import json
import re
import os
from glob import glob
import numpy as np
import torch
import random
from types import SimpleNamespace
import yaml

def extract_idx(folder_name):
    match = re.search(r'OM(\d+)$', folder_name)
    return int(match.group(1)) if match else None

def simplenamespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        return {k: simplenamespace_to_dict(v) for k, v in namespace.__dict__.items()}
    elif isinstance(namespace, list):
        return [simplenamespace_to_dict(item) for item in namespace]
    elif isinstance(namespace, dict):
        return {k: simplenamespace_to_dict(v) for k, v in namespace.items()}
    else:
        return namespace

def dict_to_simplenamespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    return d

def load_config(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return dict_to_simplenamespace(data)

def cast_array(array, dtype):                                                                                                                                                                                                                                   
    if isinstance(array, np.ndarray):                                                                                                                                                                                                                           
        return array.astype(dtype)                                                                                                                                                                                                                              
    elif isinstance(array, torch.Tensor):                                                                                                                                                                                                                       
        return array.to(dtype)                                                                                                                                                                                                                                  
    else:                                                                                                                                                                                                                                                       
        raise ValueError('Unsupported array type')  

import warnings
def load_model_checkpoint(path, model=None, optimizer=None, lr_scheduler=None, cfg=None):
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")

    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    if model is not None:
        model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    epoch = checkpoint.get('epoch', -1)
    
    print(f"Loaded checkpoint '{path}' (epoch {epoch})")
    return epoch, model, optimizer, lr_scheduler

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def numpy_to_python_native_types(obj):
    if isinstance(obj, dict):
        return {key: numpy_to_python_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_native_types(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def natural_sort(l):
    """
    Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def split_chunk_paths(chunk_dir, subject, train_val_split=0.9, shuffle=True):
    """
    Splits chunk files into train and validation sets.  
    
    Args:
    chunk_dir (str): Directory containing chunk files.
    test_subject (int): The subject number to be excluded from training and validation.
    train_val_split (float): Ratio of train to validation files.

    Returns:
    tuple: (list of train, validation, and test chunk file paths)
    """
    train_val_files = []
    test_files = []
   
    all_files = glob(os.path.join(chunk_dir, 'subject*.pkl'))
    for file in all_files:
        if f"subject_{subject}_" in file:
            test_files.append(file)
        else:
            train_val_files.append(file)

    if shuffle:            
        random.shuffle(train_val_files)  # Randomly shuffle all chunk files
    else:
        train_val_files = natural_sort(train_val_files)
    
    # Calculate the split index
    split_index = int(len(train_val_files) * train_val_split)

    # Split the files
    train_files = train_val_files[:split_index]
    val_files = train_val_files[split_index:]

    # Sort the test files in natural order
    test_files = natural_sort(test_files)
    return train_files, val_files, test_files