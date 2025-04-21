import glob
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

NUM_ENUMS = 5  # stage, p1_action, p1_character, p2_action, p2_character

# @dataclass
# class EnumColumns:
#     """Dataclass for handling enumerated columns in the dataset."""
#     stage: torch.Tensor        # Current stage/level
#     p1_action: torch.Tensor    # Player 1's current action state
#     p1_character: torch.Tensor # Player 1's character selection
#     p2_action: torch.Tensor    # Player 2's current action state
#     p2_character: torch.Tensor # Player 2's character selection

#     @classmethod
#     def from_tensor(cls, tensor: torch.Tensor, prefix: str = '') -> 'EnumColumns':
#         """Create EnumColumns from a tensor."""
#         return cls(
#             stage=tensor[..., 0],
#             p1_action=tensor[..., 1],
#             p1_character=tensor[..., 2],
#             p2_action=tensor[..., 3],
#             p2_character=tensor[..., 4]
#         )

#     def to_dict(self, prefix: str = '') -> Dict[str, torch.Tensor]:
#         """Convert EnumColumns to dictionary."""
#         return {
#             f'{prefix}stage': self.stage,
#             f'{prefix}p1_action': self.p1_action,
#             f'{prefix}p1_character': self.p1_character,
#             f'{prefix}p2_action': self.p2_action,
#             f'{prefix}p2_character': self.p2_character
#         }


@dataclass
class InputData:
    """Dataclass for handling input data."""
    continuous: torch.Tensor
    # enums: EnumColumns
    match_id: int


@dataclass
class TargetData:
    """Dataclass for handling target data."""
    continuous: torch.Tensor
    # enums: EnumColumns


class MeleeDataset(Dataset):
    """Dataset class for Super Smash Bros. Melee data."""
    
    def __init__(self, data_path: str, match_id: int, num_enums: int):
        """Initialize dataset.
        
        Args:
            data_path: Path to the .npz file containing the data
            match_id: Unique identifier for the match
            num_enums: Number of enum columns (taken from end of array)
        """
        data = np.load(data_path, allow_pickle=True)
        # Load inputs - get only continuous data
        all_inputs = data['inputs']
        # print(all_inputs.shape)
        # input_continuous = torch.tensor(all_inputs[:, :-num_enums].astype(np.float32), dtype=torch.float32)
        # input_enums = EnumColumns.from_tensor(
        #     torch.tensor(all_inputs[:, -num_enums:].astype(np.float32), dtype=torch.long)
        # )

        # FIXME: Currently the shape is off from data processing; this will fix it
        print(f"Original input shape: {all_inputs.shape}")
        
        # Transpose from (samples, features, timesteps) to (samples, timesteps, features)
        all_inputs = np.transpose(all_inputs, (0, 2, 1))
        print(f"Transposed input shape: {all_inputs.shape}")
        
        input_continuous = torch.tensor(
            all_inputs[..., :-num_enums].astype(np.float32), 
            dtype=torch.float32
        )
        self.inputs = InputData(
            continuous=input_continuous,
            # enums=input_enums,
            match_id=match_id
        )
        
        # Load targets - get only continuous data
        all_targets = data['targets']
        # target_continuous = torch.tensor(all_targets[:, :-num_enums].astype(np.float32), dtype=torch.float32)
        # target_enums = EnumColumns.from_tensor(
        #     torch.tensor(all_targets[:, -num_enums:].astype(np.float32), dtype=torch.long)
        # )
        # FIXME: Currently the shape is off from data processing; this will fix it
        all_targets = np.transpose(all_targets, (0, 2, 1))
        
        target_continuous = torch.tensor(
            all_targets[..., :-num_enums].astype(np.float32), 
            dtype=torch.float32
        )
        self.targets = TargetData(
            continuous=target_continuous,
            # enums=target_enums
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.inputs.continuous)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing input and target data
        """
        return {
            'continuous_inputs': self.inputs.continuous[idx],
            # **self.inputs.enums.to_dict(),
            'continuous_targets': self.targets.continuous[idx],
            # **self.targets.enums.to_dict(prefix='target_'),
            'match_id': self.inputs.match_id
        }


class BatchedFilesDataset:
    """Handles loading and batching of multiple dataset files."""
    
    def __init__(self, data_folder: str, files_per_batch: int = 100):
        """Initialize batched dataset loader.
        
        Args:
            data_folder: Path to folder containing .npz files
            files_per_batch: Number of files to load in each batch
        """
        self.data_folder = data_folder
        self.files_per_batch = files_per_batch
        self.num_enums = NUM_ENUMS 
        
        # Get all npz files
        self.all_files = glob.glob(f"{data_folder}/*.npz")
        self.total_files = len(self.all_files)
        self.current_batch_idx = 0
        
    def load_next_batch(self) -> ConcatDataset:
        """Load next batch of files and return as ConcatDataset."""
        start_idx = self.current_batch_idx * self.files_per_batch
        end_idx = min(start_idx + self.files_per_batch, self.total_files)
        
        # Reset if we've reached the end
        if start_idx >= self.total_files:
            self.current_batch_idx = 0
            start_idx = 0
            end_idx = min(self.files_per_batch, self.total_files)
        
        # Load the batch of files
        datasets = []
        for match_id, file_idx in enumerate(range(start_idx, end_idx)):
            file_path = self.all_files[file_idx]
            try:
                dataset = MeleeDataset(file_path, match_id, num_enums=self.num_enums)
                datasets.append(dataset)
                print(f"Loaded {file_path}")
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        
        print(datasets)
        self.current_batch_idx += 1
        return ConcatDataset(datasets)
    
    def get_all_files(self) -> List[str]:
        """Get list of all available data files.
        
        Returns:
            List of file paths
        """
        return self.all_files


def load_dataset(data_path: str, match_id: int) -> MeleeDataset:
    """Load a MeleeDataset from a file.
    
    Args:
        data_path: Path to the .npz file containing the data
        match_id: Unique identifier for the match
        
    Returns:
        Initialized MeleeDataset
    """
    return MeleeDataset(data_path, match_id)
