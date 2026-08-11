"""Utilities for loading single-coil fastMRI k-space volumes from .h5 files."""
import random
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fastmri.data import transforms as T


def load_volume_kspaces(data_dir, N_samples=20):
    """
    Load a fixed set of slices (around the middle of each volume) from all
    h5 files in a directory. Used for deterministic evaluation/test sets.

    Args:
        data_dir (str): Directory containing .h5 files.
        N_samples (int): Number of slices to pick around the middle slice
            of each volume.

    Returns:
        list of (kspace_slice: np.ndarray, slice_info: dict)
    """
    h5_files = []
    for ext in ['*.h5', '*.hdf5']:
        h5_files.extend(Path(data_dir).glob(ext))
    h5_files = sorted(h5_files)

    if len(h5_files) == 0:
        raise ValueError(f"No .h5 or .hdf5 files found in {data_dir}")

    print(f"Found {len(h5_files)} h5 files in {data_dir}")

    all_slice_counts = []
    data = []

    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as hf:
                kspace_key = None
                for key in ['kspace', 'kspace_data', 'data']:
                    if key in hf:
                        kspace_key = key
                        break

                if kspace_key is None:
                    print(f"Warning: No k-space data found in {file_path}. Available keys: {list(hf.keys())}")
                    continue

                kspace_dataset = hf[kspace_key]
                volume_shape = kspace_dataset.shape
                volume_dtype = kspace_dataset.dtype

                file_info = {
                    'filename': file_path.name,
                    'volume_shape': volume_shape,
                    'dtype': volume_dtype,
                    'attrs': dict(hf.attrs) if hasattr(hf, 'attrs') else {}
                }

                num_slices = volume_shape[0]
                all_slice_counts.append(num_slices)

                if num_slices <= N_samples:
                    slice_indices = list(range(num_slices))
                else:
                    middle_slice = num_slices // 2
                    half_samples = N_samples // 2

                    start_idx = max(0, middle_slice - half_samples)
                    end_idx = min(num_slices, middle_slice + half_samples + (N_samples % 2))

                    if end_idx - start_idx < N_samples:
                        if start_idx == 0:
                            end_idx = min(num_slices, N_samples)
                        elif end_idx == num_slices:
                            start_idx = max(0, num_slices - N_samples)

                    slice_indices = list(range(start_idx, end_idx))

                for slice_idx in slice_indices:
                    slice_kspace = kspace_dataset[slice_idx]

                    slice_info = file_info.copy()
                    slice_info['slice_index'] = int(slice_idx)
                    slice_info['slice_shape'] = slice_kspace.shape

                    data.append((slice_kspace, slice_info))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"\nExtracted {len(data)} slices from {len(h5_files)} files")

    if all_slice_counts:
        avg_slices = np.mean(all_slice_counts)
        min_slices = np.min(all_slice_counts)
        max_slices = np.max(all_slice_counts)
        total_slices = np.sum(all_slice_counts)
        print(f"\nSlice count statistics across all {len(all_slice_counts)} h5 files:")
        print(f"Average slices per file: {avg_slices:.2f}")
        print(f"Min slices per file: {min_slices}")
        print(f"Max slices per file: {max_slices}")
        print(f"Total slices across all files: {total_slices}")

    return data


class KSpaceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that samples a random h5 file and returns a k-space
    slice near the center of its volume (random offset of +/-3 slices).
    Used for training, where re-sampling per epoch is desired.

    Args:
        data_dir (str): Directory containing .h5 files.
        num_samples_per_epoch (int): Total samples per epoch. Defaults to
            the number of h5 files.
        transform (callable, optional): Optional transform applied to the
            raw k-space numpy array before conversion to a tensor.
    """

    def __init__(self, data_dir, num_samples_per_epoch=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        h5_files = []
        for ext in ['*.h5', '*.hdf5']:
            h5_files.extend(Path(data_dir).glob(ext))
        self.h5_files = sorted(h5_files)

        if len(self.h5_files) == 0:
            raise ValueError(f"No .h5 or .hdf5 files found in {data_dir}")

        if num_samples_per_epoch is None:
            num_samples_per_epoch = len(self.h5_files)

        self.num_samples_per_epoch = num_samples_per_epoch

        print(f"Found {len(self.h5_files)} h5 files in {data_dir}")
        print(f"Dataset will sample {self.num_samples_per_epoch} slices per epoch")

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        """
        Randomly selects an h5 file and extracts a slice near its center.

        Returns:
            torch.Tensor: k-space data, shape (H, W, 2) for single-coil
                or (num_coils, H, W, 2) for multi-coil.
        """
        file_path = random.choice(self.h5_files)

        try:
            with h5py.File(file_path, 'r') as hf:
                kspace_key = None
                for key in ['kspace', 'kspace_data', 'data']:
                    if key in hf:
                        kspace_key = key
                        break

                if kspace_key is None:
                    raise ValueError(f"No k-space data found in {file_path}. Available keys: {list(hf.keys())}")

                kspace_dataset = hf[kspace_key]
                num_slices = kspace_dataset.shape[0]

                central_slice_idx = num_slices // 2
                offset = np.random.randint(-3, 4)
                slice_idx = max(0, min(num_slices - 1, central_slice_idx + offset))
                kspace_slice = kspace_dataset[slice_idx]

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return self.__getitem__(idx)

        if self.transform is not None:
            kspace_slice = self.transform(kspace_slice)

        kspace_tensor = T.to_tensor(kspace_slice)
        return kspace_tensor


def collate_kspace_with_crop(batch, crop_size=(208, 208)):
    """
    Center-crops (or zero-pads) each k-space sample to `crop_size` before
    stacking into a batch, to handle scans of varying spatial dimensions.

    Args:
        batch: list of k-space tensors, shape (H, W, 2) [single-coil] or
            (num_coils, H, W, 2) [multi-coil].
        crop_size: (crop_H, crop_W) target size.
    Returns:
        Stacked batch tensor.
    """
    cropped_batch = []
    crop_H, crop_W = crop_size

    for kspace in batch:
        if kspace.ndim == 3:
            H, W, channels = kspace.shape

            start_H = max(0, (H - crop_H) // 2)
            end_H = start_H + crop_H
            start_W = max(0, (W - crop_W) // 2)
            end_W = start_W + crop_W

            if H < crop_H or W < crop_W:
                pad_H_before = max(0, (crop_H - H) // 2)
                pad_H_after = max(0, crop_H - H - pad_H_before)
                pad_W_before = max(0, (crop_W - W) // 2)
                pad_W_after = max(0, crop_W - W - pad_W_before)

                kspace_cropped = F.pad(
                    kspace,
                    (0, 0, pad_W_before, pad_W_after, pad_H_before, pad_H_after),
                    mode='constant',
                    value=0
                )
            else:
                kspace_cropped = kspace[start_H:end_H, start_W:end_W, :]

        elif kspace.ndim == 4:
            num_coils, H, W, channels = kspace.shape

            start_H = max(0, (H - crop_H) // 2)
            end_H = start_H + crop_H
            start_W = max(0, (W - crop_W) // 2)
            end_W = start_W + crop_W

            if H < crop_H or W < crop_W:
                pad_H_before = max(0, (crop_H - H) // 2)
                pad_H_after = max(0, crop_H - H - pad_H_before)
                pad_W_before = max(0, (crop_W - W) // 2)
                pad_W_after = max(0, crop_W - W - pad_W_before)

                kspace_cropped = F.pad(
                    kspace,
                    (0, 0, pad_W_before, pad_W_after, pad_H_before, pad_H_after),
                    mode='constant',
                    value=0
                )
            else:
                kspace_cropped = kspace[:, start_H:end_H, start_W:end_W, :]
        else:
            raise ValueError(
                f"Unexpected k-space tensor shape: {kspace.shape}. "
                "Expected 3D (H, W, 2) or 4D (num_coils, H, W, 2)"
            )

        cropped_batch.append(kspace_cropped)

    return torch.stack(cropped_batch, dim=0)


def create_kspace_dataloader(data_dir, batch_size, num_samples_per_epoch=None,
                              shuffle=True, num_workers=0, transform=None, crop_size=(208, 208)):
    """
    Create a DataLoader over KSpaceDataset with center-crop collation.

    Example:
        >>> train_loader = create_kspace_dataloader(
        ...     data_dir='path/to/train',
        ...     batch_size=16,
        ...     shuffle=True,
        ...     crop_size=(208, 208)
        ... )
        >>> for batch in train_loader:
        ...     print(batch.shape)  # (16, ..., 208, 208, 2)
    """
    dataset = KSpaceDataset(
        data_dir=data_dir,
        num_samples_per_epoch=num_samples_per_epoch,
        transform=transform
    )

    collate_fn = partial(collate_kspace_with_crop, crop_size=crop_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )

    return dataloader
