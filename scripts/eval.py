import argparse
import pickle
import json
import os
import re
import numpy as np
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from pressure.util.util import *
from pressure.util.selection import *
from pressure.util.stats import * 
from pressure.util.manager import create_experiment
from pressure.util.pipeline import Pipeline
from pressure.data.data_support import DataNormalizer

def create_zero_pressure_mask(pressure_batch, active_only):
    """Create a mask for frames with zero pressure."""
    if active_only:
        return torch.sum(pressure_batch, dim=1) == 0
    else:
        return torch.sum(pressure_batch.reshape(pressure_batch.shape[0], -1), dim=1) == 0

def eval_model(model, test_loader, result_save_dir, zero_conf_thresh, data_id, experiment_manager,
               cfg, logger, writer=None, global_step=0, data_root=None, run_label=None):
    """
    Evaluate model performance on test data.
    """
    device = cfg.default.device
    model_dtype = next(model.parameters()).dtype
    model = model.to(device)
    model.eval()

    # Allow overriding the data root so we can handle per-subject OM
    # directories while keeping cfg/default consistent elsewhere.
    normalizer = DataNormalizer(data_root or cfg.default.data_path, cfg=cfg)
    normalizer.verify_consistency(cfg)

    outputs = defaultdict(list)
    targets = defaultdict(list)
    frame_mask = []
    seq_len = cfg.data.sequence_length
    
    # Determine whether to reconstruct full maps for saving
    reconstruct_full = cfg.data.active_only
   
    with torch.no_grad():
        for batch in test_loader:
            # Process batch
            batch = {k: v.to(device).to(model_dtype) for k, v in batch.items()}
            batch['joint'] = batch['joint'].squeeze(2)
            if 'pressure' in batch:
                batch['pressure'] = batch['pressure'].view(-1, np.prod(batch['pressure'].size()[1:]))
            
            center_idx = seq_len // 2
            center_zero_count = torch.sum(batch['joint'][:, center_idx, :, -1] == 0, dim=-1)
            middle_valid = center_zero_count <= cfg.eval.zero_conf_thresh
            frame_valid = torch.all(batch['joint'][..., -1] != 0, dim=-1)
            non_center_frames = torch.cat([frame_valid[:, :center_idx], frame_valid[:, center_idx+1:]], dim=1)
            bad_frame_count = torch.sum(~non_center_frames, dim=1)
            other_frames_valid = bad_frame_count <= cfg.eval.bad_frames_in_seq_thresh
            if 'pressure' in batch:
                zero_pressure_mask = create_zero_pressure_mask(batch['pressure'], cfg.data.active_only)
                pressure_valid = ~zero_pressure_mask
            else:
                pressure_valid = torch.ones_like(middle_valid, dtype=torch.bool)
            mask = middle_valid & other_frames_valid & pressure_valid
            frame_mask.extend(mask.cpu().numpy())
            
            # Get model predictions
            output = model(batch['joint'])
            
            # Collect outputs and targets
            for key in output:
                outputs[key].extend(output[key].cpu().numpy())
                targets[key].extend(batch[key].cpu().numpy())
            
            # Collect middle frame joints for CoM visualization
            if 'com' in cfg.default.mode:
                outputs['middle_frame_joints'].extend(batch['middle_frame_joints'].cpu().numpy())

    # Convert lists to arrays
    for key in outputs:
        outputs[key] = np.array(outputs[key])
        targets[key] = np.array(targets[key])
   
    # Process frame mask
    frame_mask = np.array(frame_mask, dtype=np.float32)
    frame_mask[frame_mask == 0] = np.nan

    # Denormalize data before saving or calculating metrics
    if 'com' in outputs:
        outputs['com'] = normalizer.denormalize_com(outputs['com'], data_id=data_id)
        targets['com'] = normalizer.denormalize_com(targets['com'], data_id=data_id)
        outputs['middle_frame_joints'] = normalizer.denormalize_joints(outputs['middle_frame_joints'], data_id=data_id) 
    
    # Denormalize pressure data if present
    if 'pressure' in outputs:
        outputs['pressure'] = normalizer.unnormalize_and_scale_to_kpa(
            outputs['pressure'], data_id, cfg, reconstruct_full=reconstruct_full
        )
        targets['pressure'] = normalizer.unnormalize_and_scale_to_kpa(
            targets['pressure'], data_id, cfg, reconstruct_full=reconstruct_full
        )
        
    # Denormalize contact data if present
    if 'contact' in outputs:
        outputs['contact'] = normalizer.unnormalize_contact(
            outputs['contact'], data_id, cfg, reconstruct_full=False, apply_sigmoid=~cfg.data.binary_contact
        )
        targets['contact'] = normalizer.unnormalize_contact(
            targets['contact'], data_id, cfg, reconstruct_full=False
        )

    # Choose a descriptive run label for saving outputs/metrics
    if run_label is None:
        if cfg.default.om:
            run_label = f'OM_{data_id}'
        else:
            run_label = f'subject{data_id}'

    # Save outputs if enabled
    if cfg.eval.save_output:
        output_dir = os.path.join(result_save_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        output_dict = {
            'predictions': outputs,
            'targets': targets,
            'frame_mask': frame_mask,
        }
       
        if 'pressure' in outputs:
            output_dict['pressure_sums'] = np.nansum(
                targets['pressure'].reshape(len(targets['pressure']), -1), axis=1
            )

        output_path = os.path.join(output_dir, f'{run_label}_output.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
   
    # Create videos if enabled
    if cfg.viz.video.enabled:
        start_time = time.time()
        logger.info("Creating visualization videos...")
        video_dir = Path(result_save_dir) / run_label / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiment_manager.visualizer.create_modality_videos(
                predictions=outputs,
                targets=targets,
                subject=run_label,
                frame_mask=None,  # frame_mask,
            )
            logger.info(f"Visualization videos created in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to create visualization videos: {str(e)}")
            logger.exception(e)
            
    # Calculate metrics for pressure data
    metrics = {}
    for key in outputs:
        if key == 'middle_frame_joints':
            continue
        metrics[key] = {}
        if key == 'pressure':
            # Data is already in kPa
            pred_kpa = outputs['pressure']
            gt_kpa = targets['pressure']
            
            sub_weight = normalizer.get_subject_weight(data_id)
            foot_mask = np.load('assets/foot_mask_nans.npy')
            metrics['pressure'] = calc_stats(gt_kpa, pred_kpa, foot_mask, data_id, sub_weight, 
                                        frame_mask=frame_mask, writer=writer, 
                                        global_step=global_step, active_only=cfg.data.active_only)
        elif key == 'com':
            metrics['com']['l2_error_mean'] = l2_error(outputs['com'], targets['com'], frame_mask, gt_com=cfg.data.gt_com)
            metrics['com']['l2_error_median'] = l2_error(outputs['com'], targets['com'], frame_mask, gt_com=cfg.data.gt_com, metric='median')
        elif key == 'contact':
            metrics['contact']['topK'] = top_k_accuracy(outputs['contact'], targets['contact'], frame_mask, binary_contact=cfg.data.binary_contact)
        else:
            continue
    
    # Save results under an eval/ subdirectory
    eval_dir = os.path.join(result_save_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, f'{run_label}.json')

    logger.info(f'Saving results to {save_path}')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, cls=NumpyEncoder, indent=4)
  
    print_evaluation_results(metrics, logger) 
    return metrics

def main(cfg):
    """Main evaluation function."""
    # Create experiment manager
    manager = create_experiment(cfg)
    logger = manager.logger
    
    # Setup pipeline
    pipeline = Pipeline(manager)
    pipeline.setup_training('OM', skip_support=True)
    
    # Default evaluation directory (used for non-OM evaluation)
    eval_dir = manager.experiment_path / 'eval'
    eval_dir.mkdir(exist_ok=True)
    
    if cfg.default.om:
        logger.info("Evaluating Ordinary Movement (OM) data")

        data_root = cfg.default.data_path
        entries = [e for e in os.scandir(data_root) if e.is_dir()]
        has_om_dirs = any('OM' in e.name for e in entries)
        has_subject_dirs = not has_om_dirs

        # Load model checkpoint once for all OM evaluations
        checkpoint_path = Path(f'{cfg.default.checkpoint_path}/model_{cfg.eval.checkpoint_flag}.pth')
        if not checkpoint_path.exists():
            logger.warning(f'Checkpoint {checkpoint_path} does not exist')
            return

        logger.info(f'Evaluating OM data using checkpoint {checkpoint_path}')
        pipeline.load_checkpoint(checkpoint_path, model_only=True)

        # Derive the tail of the experiment path (e.g., pressure/kld/seq_...)
        exp_id = manager.exp_config.get_experiment_id(include_timestamp=False)
        exp_parts = list(exp_id.parts)
        # Drop data_type, OM flag, and model name (first three components)
        tail_parts = exp_parts[3:] if len(exp_parts) > 3 else exp_parts

        all_metrics = {}

        if has_om_dirs and not has_subject_dirs:
            # Case A: data_path directly contains OM*/ folders (single-subject OM layout)
            logger.info("Detected single-subject OM layout")
            om_folders = sorted([e.name for e in entries if 'OM' in e.name])

            for om_folder in om_folders:
                om_idx = extract_idx(om_folder)
                run_label = f'OM{om_idx}'
                logger.info(f"Processing OM {om_idx} data ...")

                # Create dataset for this OM
                _, _, test_dataset = create_dataset(cfg, subject=None, om_idx=om_idx)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.dataloader_workers
                )

                # For single-subject OM, keep using the manager's experiment path
                # but still follow the eval/output structure inside it.
                single_root = manager.experiment_path

                # Evaluate
                metrics = eval_model(
                    model=pipeline.model,
                    test_loader=test_loader,
                    result_save_dir=single_root,
                    zero_conf_thresh=cfg.eval.zero_conf_thresh,
                    data_id=f'{om_idx}',
                    experiment_manager=manager,
                    cfg=cfg,
                    writer=manager.writer,
                    logger=logger,
                    data_root=data_root,
                    run_label=run_label,
                )
                all_metrics[run_label] = metrics

        elif has_subject_dirs:
            # Case B: data_path contains multiple subject folders, each with OM*/
            logger.info("Detected multi-subject OM layout")
            subject_dirs = sorted(e.path for e in entries)
            original_data_path = cfg.default.data_path

            for subject_dir in subject_dirs:
                subject_name = os.path.basename(subject_dir)
                logger.info(f"Processing OM data for subject {subject_name}...")

                subject_entries = [e for e in os.scandir(subject_dir) if e.is_dir() and 'OM' in e.name]
                if not subject_entries:
                    logger.info(f"  No OM directories found in {subject_dir}, skipping.")
                    continue

                subject_om_folders = sorted(e.name for e in subject_entries)

                for om_folder in subject_om_folders:
                    om_idx = extract_idx(om_folder)
                    run_label = f'OM{om_idx}'
                    logger.info(f"  Processing {subject_name} OM {om_idx} data ...")

                    # Temporarily point data_path to this subject's OM chunk dir
                    cfg.default.data_path = subject_dir
                    try:
                        _, _, test_dataset = create_dataset(cfg, subject=None, om_idx=om_idx)
                    finally:
                        cfg.default.data_path = original_data_path

                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        num_workers=cfg.training.dataloader_workers
                    )

                    # Build per-subject results root:
                    # <results_path>/<data_type>/<subject_name>/pressure/kld/seq_...
                    subject_root = Path(cfg.default.results_path) / cfg.data.data_type / subject_name
                    if tail_parts:
                        subject_root = subject_root.joinpath(*tail_parts)

                    subject_root.mkdir(parents=True, exist_ok=True)

                    metrics = eval_model(
                        model=pipeline.model,
                        test_loader=test_loader,
                        result_save_dir=str(subject_root),
                        zero_conf_thresh=cfg.eval.zero_conf_thresh,
                        data_id=f'{om_idx}',
                        experiment_manager=manager,
                        cfg=cfg,
                        writer=manager.writer,
                        logger=logger,
                        data_root=subject_dir,
                        run_label=run_label,
                    )
                    # Use a composite key for metrics dict to avoid collisions
                    all_metrics[f'{subject_name}_{run_label}'] = metrics
        else:
            raise ValueError(f"Could not determine OM directory structure in {data_root}")
    else: 
        # Process each subject
        checkpoint_flag = cfg.eval.checkpoint_flag
        checkpoint_base = manager.experiment_path / 'checkpoints'
        checkpoint_paths = list(checkpoint_base.glob('Subject*'))
    
        for sub_path in checkpoint_paths:
            subject = int(re.findall(r'\d+', sub_path.name)[0])
            checkpoint = sub_path / f'model_{checkpoint_flag}.pth'
            
            if not checkpoint.exists():
                logger.warning(f'Checkpoint {checkpoint} does not exist')
                continue
                
            logger.info(f'Evaluating subject {subject} using checkpoint {checkpoint}')
           
            if cfg.data.chunk_data:
                # Create dataset and loader
                _, _, test_dataset = create_dataset(cfg, subject)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.dataloader_workers
                )
                
                # Setup pipeline and evaluate
                pipeline.setup_training(subject)
                pipeline.load_checkpoint(checkpoint)
            
                metrics = eval_model(
                    model=pipeline.model, 
                    test_loader=test_loader, 
                    result_save_dir=eval_dir, 
                    zero_conf_thresh=cfg.eval.zero_conf_thresh,
                    subject=subject, 
                    experiment_manager=manager,  # Pass the experiment manager
                    cfg=cfg, 
                    writer=manager.writer, 
                    logger=logger
                )
                all_metrics.append(metrics)
            else:
                raise NotImplementedError("Non-chunked data evaluation not supported")
   
    # Save mean metrics
    mean_metrics = collect_metrics(all_metrics)
    eval_dir = manager.experiment_path / 'eval'
    metrics_path = eval_dir / 'mean_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(mean_metrics, f, cls=NumpyEncoder, indent=4)
                 
    manager.close()
    return all_metrics, mean_metrics

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate pressure prediction model')
    argparser.add_argument('--config', type=str, default='configs/pns.yaml', 
                          help='Path to the config file')
    args = argparser.parse_args()
    
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found.")
    
    main(cfg)