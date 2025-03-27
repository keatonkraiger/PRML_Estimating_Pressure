import json
import math
import numpy as np
import torch
from torchvision.transforms import ToTensor

from pressure.data.dataset import PSUTMM100_LOSO, PSUTMM100_Temporal_LOSO_Chunked
from pressure.models.pns import PNS
from pressure.models.footformer import FootFormer
from pressure.util.util import split_chunk_paths
from pressure.util.losses import PressureLoss

data_dims = {
    'BODY25': (24, 3), # 2D openpose joints + 1 conf.  (with background removed)
    'BODY25_3D': (24, 4), # 3D openpose joints+ 1 conf. (with background removed)
    'HRNET': (17, 3), # 2D HRNET joints + 1 conf.
    'HRNET_3D': (17, 4), # 3D HRNET joints  + 1 conf.
    'MOCAP': (17, 3), # 2D MOCAP joints + 1 conf.
    'MOCAP_3D': (17, 4), # 3D MOCAP joints + 1 conf.
}
insole_shapes = {
    'full_pressure': (60, 21, 2), # Output foot pressure map
    'active_pressure': (1910) # Output of active pressure pixels only
}

def select_model(cfg, foot_mask=None):
    print(f"Creating model: {cfg.network.model}")
    input_shape = data_dims[cfg.data.data_type]
    input_size = np.prod(input_shape)
    
    # Calculate output shapes for each task
    output_shapes = {}
    output_dims = {}
    
    if 'pressure' in cfg.default.mode:
        pressure_shape = insole_shapes['active_pressure'] if cfg.data.active_only else insole_shapes['full_pressure']
        output_shapes['pressure'] = pressure_shape
        output_dims['pressure'] = np.prod(pressure_shape)
        
    if 'contact' in cfg.default.mode:
        if not cfg.data.use_regions:
            contact_shape = insole_shapes['active_pressure'] if cfg.data.active_only else insole_shapes['full_pressure']
        else:
            # For regional contact, shape is num_regions * 2 (both feet)
            contact_shape = (cfg.data.num_regions[0] * cfg.data.num_regions[1] * 2,)
        output_shapes['contact'] = contact_shape
        output_dims['contact'] = np.prod(contact_shape)
        
    if 'com' in cfg.default.mode:
        if cfg.data.gt_com:
            output_dims['com'] =4
        else:
            output_dims['com'] = input_shape[1]
 
    if cfg.network.model == 'pns':
        model = PNS(input_size=input_size, hidden_count=cfg.network.hidden_count, FC_size=cfg.network.FC_size, 
                    output_size=output_dims['pressure'], foot_mask=foot_mask, dropout=cfg.network.dropout, mult_by_mask=cfg.network.mask_mult)
    elif cfg.network.model == 'footformer':
        pred_distribution = cfg.data.pressure_is_distribution
        model = FootFormer(num_joints=input_shape[0], joint_dim=input_shape[1], pose_embed_dim=cfg.network.pose_embed_dim, seq_len=cfg.data.sequence_length,
                           num_heads=cfg.network.num_heads, num_layers=cfg.network.num_layers, output_dims=output_dims, dropout_p=cfg.network.dropout, transformer=cfg.network.transformer, pos=cfg.network.pos, 
                           mlp_dim=cfg.network.mlp_dim, pool=cfg.network.pool, decoder_dim=cfg.network.decoder_dim, mode=cfg.default.mode, pose_embedder=cfg.network.pose_embedder, pred_distribution=pred_distribution)
    else:
        raise NotImplementedError
    return model

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def select_train_support(cfg, model, dataset_size=None, processor=None):
    if cfg.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    elif cfg.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.training.lr,
            weight_decay=cfg.training.decay
        )
    else:
        raise NotImplementedError
    
    if cfg.training.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)
    elif cfg.training.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.decay)
    elif cfg.training.scheduler == 'cosine_warmup':
        # Calculate total steps
        total_steps = cfg.training.epochs * (dataset_size // cfg.training.batch_size)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=total_steps
        )

    loss_fn = PressureLoss(cfg)
    
    return optimizer, scheduler, loss_fn

def ensure_consistent_data(cfg):
    """Ensure data configuration is consistent."""
    with open(f'{cfg.default.data_path}/args.json', 'r') as f:
        data_args = json.load(f)

    if 'pressure' in cfg.default.mode:
        if cfg.data.pressure_is_distribution:
            # If pressure is a distribution, ensure other normalizations are disabled
            assert not data_args.get('max_norm', False), 'max_norm should be False when pressure is a distribution'
            assert not data_args.get('weight_norm', False), 'weight_norm should be False when pressure is a distribution'
            assert not cfg.data.subject_wise_weight_norm, 'subject_wise_weight_norm should be False when pressure is a distribution'
            assert not cfg.data.subject_wise_max_norm, 'subject_wise_max_norm should be False when pressure is a distribution'
        else:
            # Only check normalization consistency for non-distribution data
            assert data_args.get('max_norm', False) == cfg.data.subject_wise_max_norm, \
                'Inconsistent subject-wise max normalization'
            assert data_args.get('weight_norm', False) == cfg.data.subject_wise_weight_norm, \
                'Inconsistent subject-wise weight normalization'

        # Prevent double normalization conflicts
        if cfg.training.normalize is not None:
            if cfg.data.subject_wise_max_norm and cfg.training.normalize == 'max':
                raise ValueError('Cannot use both subject-wise max normalization and dataset-wide normalization')
    
def create_dataset(cfg, subject, transform=ToTensor(), om_idx=None):
    if cfg.data.chunk_data:
        chunk_dir = cfg.default.data_path
        chunk_size = cfg.data.chunk_size
        normalize = cfg.training.normalize
        train_val_split = cfg.training.train_val_split
        shuffle_data = cfg.data.shuffle_data
        active_only = cfg.data.active_only
        sequence_length = cfg.data.sequence_length
   
        if cfg.default.om:
            sequence_length = cfg.data.sequence_length
            test_dataset = PSUTMM100_Temporal_LOSO_Chunked(chunk_dir, subject, split='test', cfg=cfg, chunk_size=chunk_size, normalization=normalize, active_only=active_only, transform=transform, files=None, shuffle=False, sequence_length=sequence_length, ordinary_movement=True, om_idx=om_idx)
            return None, None, test_dataset
        else:    
            ensure_consistent_data(cfg) 
            train_paths, val_paths, test_paths = split_chunk_paths(chunk_dir, subject, train_val_split=train_val_split, shuffle=shuffle_data)

            train_dataset = PSUTMM100_Temporal_LOSO_Chunked(chunk_dir, subject=subject, split='train', chunk_size=chunk_size, normalization=normalize, 
                                                            transform=transform, files=train_paths, shuffle=shuffle_data, sequence_length=sequence_length, active_only=active_only, cfg=cfg)
            val_dataset = PSUTMM100_Temporal_LOSO_Chunked(chunk_dir, subject, split='val', chunk_size=chunk_size, normalization=normalize, 
                                                            transform=transform, files=val_paths, shuffle=shuffle_data, sequence_length=sequence_length, active_only=active_only, cfg=cfg) 
            test_dataset = PSUTMM100_Temporal_LOSO_Chunked(chunk_dir, subject, split='test', chunk_size=chunk_size, normalization=normalize, 
                                                            transform=transform, files=test_paths, shuffle=False, sequence_length=sequence_length, active_only=active_only, cfg=cfg)
    else:    
        train_dataset = PSUTMM100_LOSO(cfg.default.data_path, subject, split='Train', transform=transform)
        test_dataset = PSUTMM100_LOSO(cfg.default.data_path, subject, split='Test', transform=transform)
        val_dataset = PSUTMM100_LOSO(cfg.default.data_path, subject, split='Val', transform=transform)
        
    return train_dataset, val_dataset, test_dataset