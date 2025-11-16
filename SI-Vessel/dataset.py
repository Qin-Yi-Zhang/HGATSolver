import os
import random
import h5py
import torch
import numpy as np
import traceback
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

class FSIDataset(Dataset):
    def __init__(self,
                 hdf5_path: str,
                 window_size: int,
                 split: str,
                 split_ratios: list = [0.8, 0.1, 0.1],
                 shuffle_simulations_seed: int = 42):
        super().__init__()
        
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 dataset file not found: {hdf5_path}")
        if sum(split_ratios) != 1.0:
            raise ValueError(f"Sum of split_ratios must be 1.0, got {sum(split_ratios)}")
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}")
            
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        self.split = split
        
        with h5py.File(self.hdf5_path, 'r') as hf:
            required_meta_keys = [
                'meta/meta_index', 
                'meta/stats_fluid_mean', 'meta/stats_fluid_std', 
                'meta/stats_solid_mean', 'meta/stats_solid_std',
                'meta/stats_phys_params_mean', 'meta/stats_phys_params_std'
            ]
            for key in required_meta_keys:
                if key not in hf:
                    raise KeyError(f"Missing key in HDF5 file: {key}")

            full_meta_index = hf['meta/meta_index'][:]
            self.sim_id_map = eval(hf.attrs['sim_id_map'])
            self.stats = {
                'fluid_mean': torch.from_numpy(hf['meta/stats_fluid_mean'][:]).float(),
                'fluid_std': torch.from_numpy(hf['meta/stats_fluid_std'][:]).float(),
                'solid_elastic_mean': torch.from_numpy(hf['meta/stats_solid_mean'][:]).float(),
                'solid_elastic_std': torch.from_numpy(hf['meta/stats_solid_std'][:]).float(),
                'phys_params_mean': torch.from_numpy(hf['meta/stats_phys_params_mean'][:]).float(),
                'phys_params_std': torch.from_numpy(hf['meta/stats_phys_params_std'][:]).float(),
            }

        all_sim_indices = sorted([int(k) for k in self.sim_id_map.keys()])
        rng = random.Random(shuffle_simulations_seed)
        rng.shuffle(all_sim_indices)
        
        n_total = len(all_sim_indices)
        n_train = int(split_ratios[0] * n_total)
        n_val = int(split_ratios[1] * n_total)
        
        split_indices = {
            'train': all_sim_indices[:n_train],
            'val': all_sim_indices[n_train : n_train + n_val],
            'test': all_sim_indices[n_train + n_val:]
        }
        
        target_sim_indices = set(split_indices[self.split])
        self.meta_index = full_meta_index[np.isin(full_meta_index[:, 0], list(target_sim_indices))]
        
        if len(self.meta_index) == 0: 
            raise ValueError(f"No data found for split '{self.split}'")
        
        self._probe_feature_dims()
        self.h5_file = None

    def _probe_feature_dims(self):
        with h5py.File(self.hdf5_path, 'r') as hf:
            first_sim_idx = self.meta_index[0, 0]
            sim_grp = hf[str(first_sim_idx)]
            self.fluid_single_frame_feat_dim = sim_grp['fluid_features_ts'].shape[2]
            self.solid_elastic_single_frame_feat_dim = sim_grp['solid_elastic_features_ts'].shape[2]
            self.phys_params_dim = sim_grp['phys_params'].shape[0]

            self.fluid_input_dim = self.fluid_single_frame_feat_dim * self.window_size
            self.solid_elastic_input_dim = self.solid_elastic_single_frame_feat_dim * self.window_size
    
    def _normalize(self, tensor, domain: str):
        mean = self.stats[f'{domain}_mean'].to(tensor.device)
        std = self.stats[f'{domain}_std'].to(tensor.device)
        return (tensor - mean) / std

    def __len__(self):
        return len(self.meta_index)

    def __getitem__(self, idx):
        try:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.hdf5_path, 'r')

            sim_idx, start_frame = self.meta_index[idx]
            sim_group = self.h5_file[str(sim_idx)]
            
            k = self.window_size
            input_slice = slice(start_frame, start_frame + k)
            target_idx = start_frame + k
            
            fluid_coords = torch.from_numpy(sim_group['fluid_coords'][:]).float()
            solid_elastic_coords = torch.from_numpy(sim_group['solid_elastic_coords'][:]).float()
            solid_rigid_coords = torch.from_numpy(sim_group['solid_rigid_coords'][:]).float()
            
            data = HeteroData()
            
            data['fluid'].num_nodes = fluid_coords.shape[0]
            data['solid_elastic'].num_nodes = solid_elastic_coords.shape[0]
            data['solid_rigid'].num_nodes = solid_rigid_coords.shape[0]
            
            phys_params_raw = torch.from_numpy(sim_group['phys_params'][:]).float()
            phys_params_normalized = self._normalize(phys_params_raw, 'phys_params')

            data['fluid'].pos = fluid_coords
            fluid_feat_window = torch.from_numpy(sim_group['fluid_features_ts'][input_slice]).float()
            fluid_feat_target = torch.from_numpy(sim_group['fluid_features_ts'][target_idx]).float()
            data['fluid'].x = self._normalize(fluid_feat_window, 'fluid').permute(1, 0, 2).flatten(start_dim=1)
            data['fluid'].y = self._normalize(fluid_feat_target, 'fluid')
            data['fluid'].phys_params = phys_params_normalized.repeat(data['fluid'].num_nodes, 1)

            data['solid_elastic'].pos = solid_elastic_coords
            solid_feat_window = torch.from_numpy(sim_group['solid_elastic_features_ts'][input_slice]).float()
            solid_feat_target = torch.from_numpy(sim_group['solid_elastic_features_ts'][target_idx]).float()
            data['solid_elastic'].x = self._normalize(solid_feat_window, 'solid_elastic').permute(1, 0, 2).flatten(start_dim=1)
            data['solid_elastic'].y = self._normalize(solid_feat_target, 'solid_elastic')
            data['solid_elastic'].phys_params = phys_params_normalized.repeat(data['solid_elastic'].num_nodes, 1)

            data['solid_rigid'].pos = solid_rigid_coords
            data['solid_rigid'].phys_params = phys_params_normalized.repeat(data['solid_rigid'].num_nodes, 1)

            data['fluid', 'f2f', 'fluid'].edge_index = torch.from_numpy(sim_group['fluid_edge_index'][:]).long()
            data['solid_elastic', 'se2se', 'solid_elastic'].edge_index = torch.from_numpy(sim_group['solid_elastic_edge_index'][:]).long()
            data['solid_elastic', 'se2f', 'fluid'].edge_index = torch.from_numpy(sim_group['solid_to_fluid_edge_index'][:]).long()
            data['fluid', 'f2se', 'solid_elastic'].edge_index = torch.from_numpy(sim_group['fluid_to_solid_edge_index'][:]).long()
            data['solid_rigid', 'sr2f', 'fluid'].edge_index = torch.from_numpy(sim_group['rigid_to_fluid_edge_index'][:]).long()
            data['solid_rigid', 'sr2se', 'solid_elastic'].edge_index = torch.from_numpy(sim_group['rigid_to_elastic_edge_index'][:]).long()
            
            all_times = sim_group['all_time_steps'][:]
            dt = all_times[target_idx] - all_times[target_idx - 1] if target_idx > 0 else all_times[0]
            data.dt = torch.tensor(dt, dtype=torch.float)
            
            return data
            
        except Exception as e:
            sim_idx_at_error = self.meta_index[idx][0] if 'idx' in locals() and idx < len(self.meta_index) else -1
            print(f"Dataset error at index {idx} (sim_id: {sim_idx_at_error}). Error: {e}")
            traceback.print_exc()
            return None

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()