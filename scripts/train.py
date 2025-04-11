import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from pressure.util.util import *
from pressure.util.stats import collect_metrics
from pressure.util.manager import *
from pressure.util.pipeline import Pipeline
from pressure.util.selection import *
from scripts.eval import eval_model

def main(cfg, args=None):
    manager =  create_experiment(cfg)
    cfg = manager.config
    logger = manager.logger
    writer = manager.writer
    
    logger.info(f"Model: {cfg.network}")
    logger.info(f"Data: {cfg.data}")
    logger.info(f"Training: {cfg.training}")
    
    if cfg.default.subjects == 'all':
        subjects = np.arange(1, 11)
    else:
        subjects = cfg.default.subjects
    subjects.sort()
   
    # Setup the pipeline
    pipeline = Pipeline(manager)
    
    # Handle resuming from checkpoint
    if args is not None and args.resume:
        latest_subject, latest_checkpoint = manager.find_most_recent_checkpoint()
        if latest_subject is not None:
            # Filter subjects to start from the latest one
            subjects = [sub for sub in subjects if sub >= latest_subject]
            logger.info(f"Resuming training from subject {latest_subject}")
            logger.info(f"Remaining subjects to train: {subjects}")
        else:
            latest_checkpoint = None
            logger.info("No checkpoint found. Starting training from scratch.")
    else:
        latest_checkpoint = None

    epochs = []
    all_metrics = []
     
    for subject in subjects:
        logger.info(f'Training for subject {subject}...')
      
        # Create the dataloaders 
        train_dataset, val_dataset, test_dataset = create_dataset(cfg, subject) 
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, 
                                shuffle=False, num_workers=cfg.training.dataloader_workers)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, 
                              shuffle=False, num_workers=cfg.training.dataloader_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, 
                               shuffle=False, num_workers=cfg.training.dataloader_workers)

        # Setup pipeline for this subject
        dataset_size = len(train_dataset)
        pipeline.setup_training(subject, dataset_size=dataset_size)
        
        # Load checkpoint if resuming and it's the latest subject
        if args is not None and args.resume and latest_checkpoint is not None and subject == latest_subject:
            epoch = pipeline.load_checkpoint(latest_checkpoint)
            logger.info(f"Resuming training from epoch {epoch}...")
            args.resume = False  # Reset resume flag after loading
        else:
            logger.info(f"Starting training from scratch for subject {subject}...")
            epoch = 0

        # Rest of the training loop remains the same...
        if cfg.default.train:
            _, global_step = pipeline.train(train_loader, val_loader, subject, epoch_completed=epoch)
        else:
            global_step = 0
            
        # Load best model for evaluation
        num_epochs = pipeline.load_best_model(cfg.eval.checkpoint_flag)
        logger.info(f'Loaded best checkpoint from epoch {num_epochs} ...')
        epochs.append(num_epochs)
    
        if cfg.default.test:
            _ = pipeline.test(test_loader, subject, global_step=global_step)
            
        if cfg.default.eval: 
            eval_dir = manager.experiment_path / 'eval'
            eval_dir.mkdir(exist_ok=True)
            
            metrics = eval_model(pipeline.model, test_loader, eval_dir, 
                               zero_conf_thresh=cfg.eval.zero_conf_thresh,
                               data_id=subject, 
                               experiment_manager=manager,
                               cfg=cfg, writer=writer, logger=logger,
                               global_step=global_step)
            
            all_metrics.append(metrics)
            for metric_type, values in metrics['pressure'].items():
                writer.add_scalar(f'Metrics/Subject_{subject}/{metric_type}', 
                                values[0], global_step=global_step)
            
    manager.close()
    mean_metrics = collect_metrics(all_metrics)
    
    # Log final results
    logger.info('Training completed.')
    logger.info(f'Training for subjects {subjects} completed. Epochs: {epochs}')
    logger.info('\nMean metrics:')
    
    # Print pressure metrics
    if 'pressure' in mean_metrics:
        logger.info('\nPressure:')
        for metric_name, value in mean_metrics['pressure'].items():
            logger.info(f'    {metric_name}: {value:.3f}')
    
    # Print COM metrics
    if 'com' in mean_metrics:
        logger.info('\nCoM:')
        for metric_name, value in mean_metrics['com'].items():
            logger.info(f'    {metric_name}: {value:.3f}')
    
    # Print contact metrics
    if 'contact' in mean_metrics:
        logger.info('\nContact:')
        for k, value in mean_metrics['contact']['topK'].items():
            logger.info(f'    Top-{k} Accuracy: {value:.3f}')
            
    
    # Save mean metrics
    eval_dir = manager.experiment_path / 'eval'
    metrics_path = eval_dir / 'mean_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(mean_metrics, f, cls=NumpyEncoder, indent=4)
    
    return mean_metrics
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train a model for pressure prediction')
    argparser.add_argument('--config', type=str, default='configs/pns.yaml', help='Path to the config file')
    argparser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint') 
    args = argparser.parse_args() 
   
    try:
        cfg = load_config(args.config)
        main(cfg, args)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {args.config}")
        