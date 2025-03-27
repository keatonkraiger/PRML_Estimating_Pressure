from glob import glob
import json
import json
import pickle
import h5py
import random
import re
import numpy as np
import os

import torch
from torch.utils.data import Dataset

from pressure.util import *
from pressure.util.util import extract_idx
from pressure.data.data_support import ContactMapConfig, PressureMapProcessor 
    
class PSUTMM100_Temporal_LOSO_Chunked(Dataset):
    """
    Creates a LOSO PSUTMM100 dataset. Given a subject number, the dataset will
    load the data of all other subjects as training data and the data of the 
    given subject as test data.
    """
    def __init__(self, chunk_dir, subject=1, split='train', chunk_size=5000, normalization='max', files=None, shuffle=True, 
                 sequence_length=5, transform=None, ordinary_movement=False, active_only=False, cfg=None, om_idx=None):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.normalization = normalization
        self.transform = transform
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.subject = subject
        self.ordinary_movement = ordinary_movement
        self.split = split
        self.mode = cfg.default.mode 
        self.active_only = active_only
       
        if ordinary_movement:
            if om_idx is None:
                om_idx = extract_idx(chunk_dir)
            metadata = json.load(open(f"{chunk_dir}/OM{om_idx}/OM_{om_idx}_metadata.json", 'r'))
            self.metadata = [item for item in metadata]
            self.chunk_files = [item['file_path'] for item in metadata]
        else: 
            metadata = json.load(open(f"{chunk_dir}/subject_{subject}_metadata.json", 'r')) if split == 'test'  \
                else [json.load(open(f"{chunk_dir}/subject_{i}_metadata.json", 'r')) for i in range(1, 11) if i != subject]
           
            if split != 'test': 
                metadata = [item for sublist in metadata for item in sublist]

            if files is not None:
                self.chunk_files = files 
                self.metadata = []
                self.metadata = [item for item in metadata if item['file_path'] in files]
            else:
                raise NotImplementedError 
            
        self.loaded_files = set()
        self.chunk_cache = {}
        
        # Generate sequence start indices
        self.sequence_starts = self.generate_sequence_starts()
        if self.shuffle:
            random.shuffle(self.sequence_starts)
        
        if self.active_only:
            foot_mask = np.load('assets/foot_mask_nans.npy') # (60,21,2)
            self.active_indices = np.where(foot_mask.flatten() == 1)[0] 
            
        if 'contact' in self.mode:
            contact_config = ContactMapConfig(
                use_regions=cfg.data.use_regions,
                contact_threshold=cfg.data.contact_threshold,
                num_regions=cfg.data.num_regions,
                active_only=self.active_only,
                binary_contact=cfg.data.binary_contact
            )
            self.processor = PressureMapProcessor(contact_config)
            
    def get_global_max_pressure(self, chunk_dir):
        with open(f"{chunk_dir}/foot_pressure_max.pkl", 'rb') as f:
            pressure_dict = pickle.load(f)
        
        if self.ordinary_movement:
            max_foot_pressure = pressure_dict
        else:
            max_foot_pressure = pressure_dict['max_foot_pressure']
            subject_ids = pressure_dict['subject_ids']
            current_subject_idx = subject_ids.index(self.subject)
            # remove the current subject from the list of max pressures
            current_subject_idx = subject_ids.index(self.subject)
            del max_foot_pressure[current_subject_idx]
        return np.max(max_foot_pressure)

    def generate_sequence_starts(self):
        return list(range(0, self.__len__()))
    
    def __len__(self):
        return sum(item['chunk_size'] for item in self.metadata)
        
    def load_chunk(self, chunk_idx):
        if chunk_idx not in self.chunk_cache:
            with open(self.chunk_files[chunk_idx], 'rb') as f:
                chunk = pickle.load(f)
            if self.normalization:
                chunk = self.normalize(chunk)
            self.chunk_cache[chunk_idx] = chunk
            if self.chunk_files[chunk_idx] not in self.loaded_files:
                self.loaded_files.add(self.chunk_files[chunk_idx])
        return self.chunk_cache[chunk_idx]

    def __getitem__(self, idx):
        # Calculate the half sequence length
        half_seq = (self.sequence_length - 1) // 2

        if self.shuffle:
            # Get the middle frame index from the precomputed sequence starts for shuffled data
            middle_frame_idx = self.sequence_starts[idx]
        else:
            # Directly use the provided index as the middle frame for non-shuffled data
            middle_frame_idx = idx 
        
        # Calculate the actual start and end indices of the sequence
        start_idx = middle_frame_idx - half_seq
        end_idx = start_idx + self.sequence_length
        # Ensure start and end indices are within bounds
        end_idx = min(end_idx, self.__len__())
        frames, press, coms = [], [], []
         
        for i in range(start_idx, end_idx):
            if i < 0 or i > self.__len__():
                if i < 0:
                    i = 0
                    chunk_idx = i // self.chunk_size
                    chunk = self.load_chunk(chunk_idx)
                    idx_in_chunk = i % min(self.chunk_size, len(chunk))
                    sample = chunk[idx_in_chunk]
                else:
                    # Pad the end of the sequence with the last frame
                    i = self.__len__() - 1
                    chunk_idx = i // self.chunk_size
                    chunk = self.load_chunk(chunk_idx)
                    idx_in_chunk = i % min(self.chunk_size, len(chunk))
                    sample = chunk[idx_in_chunk]
            else:
                chunk_idx = i // self.chunk_size
                chunk = self.load_chunk(chunk_idx)
                idx_in_chunk = i % min(self.chunk_size, len(chunk))
                sample = chunk[idx_in_chunk]
            
            joint, pressure, com = sample['joint'], sample['pressure'], sample['com'] 
            # Apply transformations if any
            if self.transform:
                joint = self.transform(joint)
                pressure = self.transform(pressure)
                if com.ndim == 1:
                    # Expand dim to 2d
                    com = np.expand_dims(com, axis=0)
                com = self.transform(com).squeeze(0)
                if pressure.dim() == 3:
                    pressure = pressure.permute(1, 2, 0)
                    
            if self.active_only:
                pressure = pressure.reshape(-1)[self.active_indices]
        
            frames.append(joint)
            press.append(pressure)
            coms.append(com)

        if len(frames) != self.sequence_length:
            # At the end of the dataset, pad the sequence with the last frame
            frames.extend([frames[-1]] * (self.sequence_length - len(frames)))
            press.extend([press[-1]] * (self.sequence_length - len(press)))
            coms.extend([coms[-1]] * (self.sequence_length - len(coms)))
            
        # Ensure that the sequence is of the correct length
        assert len(frames) == self.sequence_length
      
        # Return seq of joints and then the middle frame of pressure
        joints = torch.stack(frames)
        press= torch.stack(press)
        coms = torch.stack(coms)
        
        pressure =  press[len(press)//2]
        com = coms[len(coms)//2].squeeze(0)
        middle_frame_joints = joints[len(joints)//2].squeeze(0)
        
        return self.create_return(joints, pressure, com, middle_frame_joints)    
   
    def create_return(self, joints, pressure, com, middle_frame_joints):
        result = {'joint': joints, 'middle_frame_joints': middle_frame_joints}
     
        if 'pressure' in self.mode:
            result['pressure'] = pressure
        if 'contact' in self.mode:
            processed = self.processor.process_pressure_map(pressure)
            result['contact'] = processed['contact']
        if 'com' in self.mode:
            result['com'] = com
        return result

    def create_contact_mask(self, pressure_map):
        """
        Create binary contact mask.
        """
        binary_mask = (pressure_map > self.contact_threshold).astype(np.float32)
        return binary_mask.flatten()

    def create_region_contact(self, contact_map):
        """
        Create region-based contact map.
        """
        if not self.use_regions:
            return contact_map

        contact_map = contact_map.reshape(self.original_shape)
        reshaped = contact_map.reshape(self.num_regions[0], self.region_shape[0], 
                                       self.num_regions[1], self.region_shape[1])
        region_contact = (reshaped.mean(axis=(1, 3)) > self.contact_threshold).astype(np.float32)
        return region_contact.flatten()
     
    def normalize(self, data):
        joint_data = np.array([sample[0] for sample in data])
        pressure_data = np.array([sample[1] for sample in data])
        pressure_data = np.nan_to_num(pressure_data)
        if self.normalization == 'z':
            mean = np.mean(pressure_data)
            std = np.std(pressure_data)
            np.subtract(pressure_data, mean, out=pressure_data)
            np.divide(pressure_data, std, out=pressure_data)
        elif self.normalization == 'max':
            np.divide(pressure_data, self.max_pressure, out=pressure_data)
        elif self.normalization == 'minmax':
            pressure_data  = (pressure_data - np.min(pressure_data)) / (np.max(pressure_data) - np.min(pressure_data))
            pressure_data[pressure_data == np.inf] = 0
            pressure_data = np.nan_to_num(pressure_data)

        normalized_chunk = list(zip(joint_data, pressure_data))
        return normalized_chunk 
   
class PSUTMM100_LOSO(Dataset):
    """
    Creates a LOSO PSUTMM100 dataset. Given a subject number, the dataset will
    load the data of all other subjects as training data and the data of the 
    given subject as test data.
    """
    def __init__(self, root_dir, subject, joint_type='OpenPose', view=1, sample_rate=100,
                split='Test', normalization='z', transform=None):
        self.root_dir = root_dir
        self.subject = subject
        self.joint_type = joint_type
        self.view = view
        self.sample_rate = sample_rate
        self.transform = transform
        self.split = split
        self.normalization = normalization

        self.joint_data, self.pressure_data =  self.create_dataset()
        self.input_size = self.pressure_data.shape[1]
        self.output_size = self.joint_data.shape[1]
        
    def normalize(self, data):
        # Set any nan values to 0
        data = np.nan_to_num(data)

        if self.normalization == 'z':
            mean = np.mean(data)
            std = np.std(data)
            np.subtract(data, mean, out=data)
            np.divide(data, std, out=data)
        elif self.normalization == 'max':
            np.divide(data, np.max(data), out=data)
        elif self.normalization == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data) - data_min
            np.subtract(data, data_min, out=data)
            np.divide(data, data_max, out=data)
        else:
            raise NotImplementedError

        return data


    def create_dataset(self):
        all_pressure_data = []  
        all_joint_data = []  

        # Load all subjects
        for subject in range(1, 11):
            if (self.split == 'Train' and subject == self.subject) or (self.split == 'Test' and subject != self.subject):
                continue
            subject_dir = os.path.join(self.root_dir, 'Subject'+str(subject))

            # Load data from each take for the given subject
            pressure_files = glob(os.path.join(subject_dir, 'Pressure_*.mat'))
            joint_files = glob(os.path.join(subject_dir, f'BODY25_V{self.view}_*.mat'))
            natural_numerical_sort = lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', os.path.splitext(os.path.basename(x))[0])]
            pressure_files.sort(key=natural_numerical_sort)
            joint_files.sort(key=natural_numerical_sort)

            for pressure_file, joint_file in zip(pressure_files, joint_files):
                with h5py.File(pressure_file, 'r') as f:
                    pressure_data = np.array(f['PRESSURE'])
                with h5py.File(joint_file, 'r') as f:
                    joint_data = np.array(f['POSE'])
                pressure_data = np.transpose(pressure_data, (3, 0, 1, 2))
                joint_data = np.transpose(joint_data, (2, 0, 1))

                pressure_data = pressure_data[::self.sample_rate]
                joint_data = joint_data[::self.sample_rate] 

                all_pressure_data.append(pressure_data)
                all_joint_data.append(joint_data)
        pressure_dataset = np.concatenate(all_pressure_data, axis=0)
        joint_dataset = np.concatenate(all_joint_data, axis=0)
        del all_pressure_data, all_joint_data

        # Normalize data
        pressure_dataset = self.normalize(pressure_dataset)

        return joint_dataset, pressure_dataset

    def __len__(self):
        return len(self.pressure_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pressure_sample = self.pressure_data[idx]
        joint_sample = self.joint_data[idx]

        if self.transform:
            pressure_sample = self.transform(pressure_sample)
            joint_sample = self.transform(joint_sample)

        return joint_sample, pressure_sample 
    
class PSUTMM100_LOSO_Chunked(Dataset):
    """
    Creates a LOSO PSUTMM100 dataset. Given a subject number, the dataset will
    load the data of all other subjects as training data and the data of the 
    given subject as test data.
    """
    def __init__(self, chunk_dir, subject=1, split='train', chunk_size=5000, normalization='max', zero_conf_thresh=0, files=None, shuffle=True, transform=None, max_foot_pressure=None):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.normalization = normalization
        self.transform = transform
        self.shuffle = shuffle
        self.subject = subject
       
        metadata = json.load(open(f"{chunk_dir}/subject_{subject}_metadata.json", 'r')) if split == 'test'  \
            else [json.load(open(f"{chunk_dir}/subject_{i}_metadata.json", 'r')) for i in range(1, 11) if i != subject]

        if split != 'test': 
            metadata = [item for sublist in metadata for item in sublist]

        if files is not None:
            self.chunk_files = files 
            self.metadata = []
            self.metadata = [item for item in metadata if item['file_path'] in files]
        else:
            raise NotImplementedError 
        self.loaded_files = set()
        self.chunk_cache = {}
       
        if max_foot_pressure is None: 
            self.max_pressure = self.get_global_max_pressure(chunk_dir)
        else:
            self.max_pressure = max_foot_pressure
    
    def get_global_max_pressure(self, chunk_dir):
        with open(f"{chunk_dir}/foot_pressure_max.pkl", 'rb') as f:
            pressure_dict = pickle.load(f)
        max_foot_pressure = pressure_dict['max_foot_pressure']
        subject_ids = pressure_dict['subject_ids']
        current_subject_idx = subject_ids.index(self.subject)
        # remove the current subject from the list of max pressures
        current_subject_idx = subject_ids.index(self.subject)
        del max_foot_pressure[current_subject_idx]
        return np.max(max_foot_pressure)
        
    def __len__(self):
        return sum(item['chunk_size'] for item in self.metadata)
        
    def load_chunk(self, chunk_idx):
        if chunk_idx not in self.chunk_cache:
            with open(self.chunk_files[chunk_idx], 'rb') as f:
                chunk = pickle.load(f)
            if self.shuffle:
                random.shuffle(chunk)
            if self.normalization:
                chunk = self.normalize(chunk)
            self.chunk_cache[chunk_idx] = chunk
            if self.chunk_files[chunk_idx] not in self.loaded_files:
                self.loaded_files.add(self.chunk_files[chunk_idx])
        return self.chunk_cache[chunk_idx]

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        idx_in_chunk = idx % min(self.chunk_size, len(self.load_chunk(chunk_idx)))
        chunk = self.load_chunk(chunk_idx)

        joint, pressure = chunk[idx_in_chunk]
        if self.transform:
            joint = self.transform(joint)
            pressure = self.transform(pressure)
            pressure = pressure.permute(1, 2, 0)

        return joint, pressure

    def normalize(self, data):
        joint_data = np.array([sample[0] for sample in data])
        pressure_data = np.array([sample[1] for sample in data])
        pressure_data = np.nan_to_num(pressure_data)
        if self.normalization == 'z':
            mean = np.mean(pressure_data)
            std = np.std(pressure_data)
            np.subtract(pressure_data, mean, out=pressure_data)
            np.divide(pressure_data, std, out=pressure_data)
        elif self.normalization == 'max':
            np.divide(pressure_data, self.max_pressure, out=pressure_data)
        elif self.normalization == 'minmax':
            pressure_data  = (pressure_data - np.min(pressure_data)) / (np.max(pressure_data) - np.min(pressure_data))
            pressure_data[pressure_data == np.inf] = 0
            pressure_data = np.nan_to_num(pressure_data)

        normalized_chunk = list(zip(joint_data, pressure_data))
        return normalized_chunk 