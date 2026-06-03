from typing import Optional, Callable
import sys, os
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

class CIFAR100Dataset(Dataset):
    """
    Simple CIFAR-100 dataset wrapper for use with PyTorch DataLoader.
    Loads data using prepare_dataset.py's get_dataset() function.
    """
    def __init__(
        self,
        data,
        labels,
        task='multiclass',
        return_weights=False,
        return_key=False,
        preprocess_data: Optional[Callable] = None,
        preprocess_targ: Optional[Callable] = None,
        flatten: bool = True,
    ):
        super(CIFAR100Dataset, self).__init__()
        
        self.task = task
        self.return_weights = return_weights
        self.return_key = return_key
        self.preprocess_data = preprocess_data
        self.preprocess_targ = preprocess_targ
        self.flatten = flatten
        self.data, self.labels = data, torch.as_tensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def _fetch_data(self, idx):
        data = []
        for _idx in idx:
            data.append(np.asarray(self.data[_idx]))
            
        return np.stack(data), self.labels[idx]
    
    def __getitem__(self, index):
        return self.__getitems__([index])[0]
    
    def __getitems__(self, idx):
        data, label = self._fetch_data(idx)
        
        # Apply preprocessing if provided (legacy support)
        if self.preprocess_data is not None:
            # preprocess_data expects numpy array
            data = self.preprocess_data(images = data)
            data = torch.as_tensor(data).float()
        
        # Flatten if needed (after transforms)
        if self.flatten and data.dim() > 1:
            data = data.flatten(start_dim=1)
        
        if self.preprocess_targ is not None:
            label = self.preprocess_targ(label)
        
        # Handle different task types
        if self.task == 'multiclass':
            label = label.long()
        elif self.task == 'binary':
            label = label.float().unsqueeze(-1) if label.dim() == 1 else label.float()
        elif self.task == 'multilabel':
            label = torch.nn.functional.one_hot(label.long(), 100).float()
        
        if self.return_weights:
            # For CIFAR-100, all samples have equal weight
            weight = torch.tensor(1.0, dtype=torch.float32)
        
        out = []
        for _iter, (_idx, _data, _label) in enumerate(zip(idx, data, label)):
            out_args = (_data, _label)
            
            if self.return_weights:
                out_args += weight[_iter],
                
            if self.return_key:
                out_args += _idx,
            out.append(out_args)
            
        return out


class IterableCIFAR100Dataset(IterableDataset):
    """
    Simple CIFAR-100 dataset wrapper for use with PyTorch DataLoader.
    Loads data using prepare_dataset.py's get_dataset() function.
    """
    def __init__(
        self,
        data,
        labels,
        task='multiclass',
        return_weights=False,
        return_key=False,
        preprocess_data: Optional[Callable] = None,
        preprocess_targ: Optional[Callable] = None,
        flatten: bool = True,
    ):
        super(IterableCIFAR100Dataset, self).__init__()
        
        self.task = task
        self.return_weights = return_weights
        self.return_key = return_key
        self.preprocess_data = preprocess_data
        self.preprocess_targ = preprocess_targ
        self.flatten = flatten
        self.data, self.labels = data, torch.as_tensor(labels)
        self._allowed_idx = torch.arange(len(labels))
    
    def allowed_labels(self, labels):
        self._allowed_labels = torch.tensor(list(set(labels))).to(self.labels.dtype)
        self._allowed_idx = torch.arange(len(self.labels))[
            torch.isin(self.labels, self._allowed_labels)
        ]
        self.shuffle()
        
    def shuffle(self):
        self._allowed_idx = self._allowed_idx[
            torch.randperm(self._allowed_idx.shape[0])
        ]
        
    def single_iter(self, idx):
        return (
            _ for _ in self.__getitems__(idx)
        )
        
    def __iter__(self):
        self.worker_info = get_worker_info()
        if self.worker_info is None:
            return self.single_iter(self._allowed_idx)
        else:
            slice_len = len(self) // self.worker_info.num_workers
            id = self.worker_info.id
            if id == self.worker_info.num_workers-1:
                return self.single_iter(
                    self._allowed_idx[slice_len*id:]
                )
            return self.single_iter(
                self._allowed_idx[slice_len*id:slice_len*(id+1)]
            )
    
    def __len__(self):
        return len(self._allowed_idx)
    
    def _fetch_data(self, idx):
        data = []
        for _idx in idx:
            data.append(np.asarray(self.data[_idx]))
            
        return np.stack(data), self.labels[idx]
    
    def __getitem__(self, index):
        return self.__getitems__([index])[0]
    
    def __getitems__(self, idx):
        data, label = self._fetch_data(idx)
        
        # Apply preprocessing if provided (legacy support)
        if self.preprocess_data is not None:
            # preprocess_data expects numpy array
            data = self.preprocess_data(images = data)
            data = torch.as_tensor(data).float()
        
        # Flatten if needed (after transforms)
        if self.flatten and data.dim() > 1:
            data = data.flatten(start_dim=1)
        
        if self.preprocess_targ is not None:
            label = self.preprocess_targ(label)
        
        # Handle different task types
        if self.task == 'multiclass':
            label = label.long()
        elif self.task == 'binary':
            label = label.float().unsqueeze(-1) if label.dim() == 1 else label.float()
        elif self.task == 'multilabel':
            label = torch.nn.functional.one_hot(label.long(), 100).float()
        
        if self.return_weights:
            # For CIFAR-100, all samples have equal weight
            weight = torch.tensor(1.0, dtype=torch.float32)
        
        out = []
        for _iter, (_idx, _data, _label) in enumerate(zip(idx, data, label)):
            out_args = (_data, _label)
            
            if self.return_weights:
                out_args += weight[_iter],
                
            if self.return_key:
                out_args += _idx,
            out.append(out_args)
            
        return out


if __name__ == '__main__':
    import albumentations as A
    import matplotlib.pyplot as plt
    
    from prepare_dataset import build_dataset, get_dataset
    
    # Build dataset first
    build_dataset()
    
    # Test loading train dataset
    print("Testing CIFAR100Dataset...")
    data, labels = get_dataset('train_val')
    
    preprocess_data = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(scale_limit=(-0.2,0),rotate_limit=(-90,90), p=1),
            # A.Normalize(normalization = 'min_max'), #0,1 normalization
            A.Normalize(normalization = 'standard'), #ImageNet mean/std
            # A.ToTensorV2(),    
        ], 
        telemetry   = False,
    )
    train_dataset = CIFAR100Dataset(
        data, labels,
        preprocess_data= lambda **kwargs: preprocess_data(**kwargs)[list(kwargs.keys())[0]],
        flatten=False,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    data, label = train_dataset[np.random.randint(0,len(train_dataset))]
    print(f"Data shape: {data.shape}")
    print(f"Label: {label}")
    print(f"Data dtype: {data.dtype}")
    print(f"Label dtype: {label.dtype}")
    print('Data min :', data.min())
    print('Data max :', data.max())
    
    fname = os.path.join(os.path.dirname(__file__),'dataset','sample.png')
    print('Sample image :', fname)
    plt.imsave(fname, (data - data.min()) / (data.max() - data.min()))
    
    print("\nDataset ready for training!")
