import datetime
import os

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]
            
def extract_wandb_tags_notes(cfg):
    tags = [
        f"model:{cfg.network.model}",
        f"sequence_length:{cfg.data.sequence_length}",
        f"batch_size:{cfg.training.batch_size}",
        f"lr:{cfg.training.lr}",
        f"scheduler:{cfg.training.scheduler}",
        f"optimizer:{cfg.training.optimizer}",
        f"loss:{cfg.training.loss}",
    ]
    
    notes = (
        f"Training with model {cfg.network.model}, "
        f"sequence length {cfg.data.sequence_length}, "
        f"batch size {cfg.training.batch_size}, "
        f"learning rate {cfg.training.lr}, "
        f"scheduler {cfg.training.scheduler}, "
        f"optimizer {cfg.training.optimizer}, "
        f"loss function {cfg.training}."
    )
    
    return tags, notes

def reset_wandb_env():
    exclude_keys = {"WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"}
    for key in list(os.environ):
        if key.startswith("WANDB_") and key not in exclude_keys:
            del os.environ[key]

def create_run_name(subject, config):
    """
    Creates a unique run name that includes the subject index and the current time.

    Args:
    subject (str): The subject identifier, e.g., 'Subject_1'.
    config (dict): The current sweep configuration.

    Returns:
    str: A unique run name.
    """
    # Get the current time to append to the run name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{subject}_{timestamp}"
    return run_name

def create_config_identifier(config):
    """
    Creates a unique identifier string from the current sweep configuration.
    
    Args:
    config (dict): Configuration dictionary containing hyperparameters.

    Returns:
    str: A unique identifier string for the current configuration.
    """
    # List of hyperparameter names you want to include in the identifier
    parts = []
    for param in config.keys():
        parts.append(f'{param}_{config[param]}')

    identifier = "_".join(parts)
    return identifier

def init_wandb_run(cfg, exp_name, cur_sub):
    reset_wandb_env()
    wand_cfg = wandb_cfg(cfg)
    tags, notes = extract_wandb_tags_notes(cfg)
    name = create_config_identifier(cfg)#f"{cur_sub}_{exp_name}"
    wandb.init(
        project="pressure",
        name=name,  # e.g., 'Subject_01', 'Subject_02', etc.
        config=wand_cfg,
        group=cur_sub,
        tags=tags,
        notes=notes,
        reinit=True
    )
     
def wandb_cfg(cfg):
    # Create a cleaner version of the config for logging to wandb
    wandb_cfg = {
        'model': cfg.network.model,
        'sequence_length': cfg.data.sequence_length,
        'lr': cfg.training.lr,
        'batch_size': cfg.training.batch_size,
    }
    if cfg.network.model == 'pns':
        wandb_cfg['hidden_count'] = cfg.networ.hidden_count
        wandb_cfg['FC_size'] = cfg.network.FC_size
        wandb_cfg['dropout'] = cfg.network.dropout
        wandb_cfg['mask_mult'] = cfg.network.mask_mult
    elif cfg.network.model == 'footformer':
        wandb_cfg['temporal_out_channels'] = cfg.network.temporal_out_channels
        wandb_cfg['num_heads'] = cfg.network.num_heads
        wandb_cfg['num_layers'] = cfg.network.num_layers
        wandb_cfg['dropout'] = cfg.network.dropout
        wandb_cfg['mask_mult'] = cfg.network.mask_mult 
    elif cfg.network.model == 'diffusion':
        wandb_cfg['temporal_out_channels'] = cfg.network.temporal_out_channels
        wandb_cfg['embedding_dim'] = cfg.network.embedding_dim
        wandb_cfg['num_steps'] = cfg.network.num_steps
    return wandb_cfg  