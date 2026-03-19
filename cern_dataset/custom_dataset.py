from typing import Callable, Literal
import torch
import tables
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class DistributedH5Dataset (IterableDataset):
    def __init__(
        self, 
        h5files, 
        buffer_size,
        input_cols  : list[str], 
        output_cols : list[str], 
        key_col     : str       = None,
        task        : Literal['binary','as_binary','multiclass','multilabel'] = 'multiclass',
        remove_mass_pt_window   : dict[tuple[float,float]] = False, 
        return_weights : str    = False,
        return_key              = False,
        preprocess_data : Callable[[np.ndarray], np.ndarray] = lambda x: x,
        preprocess_targ : Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ):
        super(DistributedH5Dataset, self).__init__()
        self.set_task(task)
        self.buffer_size     = buffer_size
        self.h5files         = h5files
        self.input_cols      = input_cols
        if self.task == 'binary' and len(output_cols) not in (1, 2):
            raise ValueError(f'Binary task requires one output column; got {len(output_cols)}')
        elif self.task == 'as_binary' and len(output_cols) != 2:
            raise ValueError(f'Binary task requires one output column; got {len(output_cols)}')
        self.output_cols     = output_cols
        self.key_col         = key_col
        self.return_weights  = return_weights
        self._return_key     = return_key
        self._return_weights = return_weights 
        self.preprocess_data = preprocess_data if preprocess_data is not None else lambda x: x
        self.preprocess_targ = preprocess_targ if preprocess_targ is not None else lambda x: x
        self.remove_mass_pt_window= remove_mass_pt_window
        self.total_len = None
        
    def set_task(self, task):
        if task not in ('binary','as_binary','multiclass','multilabel'):
            raise NotImplementedError(f'Task type "{task}" is not implemented.')
        self.task = task
        
    def __len__(self):
        if self.total_len is None:
            self.total_len = 0
            for file in self.h5files:
                with tables.open_file(file, 'r') as h5file:
                    self.total_len += getattr(h5file.root,self.input_cols[0]).shape[0]
        return self.total_len
        
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    def _update_buffer(self):
        # Identify current file
        if self.curr_len == 0 or self.idx_buffer >= self.curr_len :
            if  self.idx_file  == self.curr_len == 0:
                self.idx_file   = 0
                self.idx_total  = 0
            else :
                self.idx_file  += 1
            self.idx_buffer = 0
            self.curr_len   = 0
            
        if self.idx_file >= len(self.h5workerfiles):
            if hasattr(self,'data_buffer'):
                del self.data_buffer
            if hasattr(self,'targ_buffer'):
                del self.targ_buffer
            if hasattr(self,'key_buffer'):
                del self.key_buffer
            raise StopIteration
        
        file = self.h5workerfiles[self.idx_file]
        with tables.open_file(file, 'r') as h5file:
            if self.curr_len == 0:
                self.curr_len = getattr(h5file.root, self.input_cols[0])[:].shape[0]
                
            self.data_buffer = torch.tensor(
                self.preprocess_data(
                    np.stack([
                        getattr(h5file.root, col)[self.idx_buffer:self.idx_buffer+self.buffer_size]
                            for col in self.input_cols
                    ], axis = 1
                ))).float()
            self.targ_buffer = []
            for label in self.output_cols:
                if '*' in label:
                    prods = label.split('*')
                    prod0 = prods[0]
                    prod1 = prods[1]
                    fact0 = getattr(h5file.root,prod0)[self.idx_buffer:self.idx_buffer+self.buffer_size]
                    fact1 = getattr(h5file.root,prod1)[self.idx_buffer:self.idx_buffer+self.buffer_size]
                    self.targ_buffer.append(np.multiply(fact0,fact1))
                    del fact0, fact1
                else :
                    self.targ_buffer.append(
                        getattr(h5file.root,label)[self.idx_buffer:self.idx_buffer+self.buffer_size]
                    )
            
            self.targ_buffer = torch.tensor(
                self.preprocess_targ(
                    np.stack(self.targ_buffer, axis = 1)
            )).float()
                
            if self._return_weights:
                self.wght_buffer = torch.tensor(
                    getattr(h5file.root, self._return_weights)[self.idx_buffer:self.idx_buffer+self.buffer_size]
                ).float()
            if self._return_key and self.key_col is not None:
                self.key_buffer = getattr(h5file.root, self.key_col)[self.idx_buffer:self.idx_buffer+self.buffer_size]

            data_parced = len(self.data_buffer)
            
            if self.task in ('as_binary','multiclass'):
                cond = self.targ_buffer.sum(1) == 1
                self.targ_buffer = self.targ_buffer.argmax(1)
                
                if self.task == 'as_binary':
                    self.targ_buffer = self.targ_buffer.float()
            else :
                cond = torch.ones_like(self.targ_buffer[:,0]).to(torch.bool)

            if self.remove_mass_pt_window:
                spec = torch.tensor(
                        np.stack([
                            getattr(h5file.root, col)[self.idx_buffer:self.idx_buffer+self.buffer_size]
                                for col in self.remove_mass_pt_window.keys()
                        ], axis = 0
                )).float()
                cond = torch.stack([cond] + [
                    cond_i 
                        for (low_lim, high_lim), spec_i in zip(
                            self.remove_mass_pt_window.values(),
                            spec,
                        )
                        for cond_i in [
                            low_lim < spec_i,
                            spec_i  < high_lim 
                        ]  
                ], dim = -1
                ).all(-1)
                del spec
        
        self.data_buffer = self.data_buffer[cond]
        self.targ_buffer = self.targ_buffer[cond]
        if self._return_weights:
            self.wght_buffer = self.wght_buffer[cond]
        if self._return_key and self.key_col is not None:
            self.key_buffer = self.key_buffer[cond]
            
        self.idx_buffer += data_parced
        # print('Dataset :', data_parced, cond.sum().item(), self.idx_buffer, self.idx_file, self.idx)
        
    def __next__(self):
        while not hasattr(self,'data_buffer') or self.idx >= len(self.data_buffer):
            self._update_buffer()
            self.idx = 0
        
        idx = self.idx
        self.idx += 1
        self.idx_total += 1
        
        extra_args = ()
        if self._return_weights:
            extra_args += self.wght_buffer[idx],
            
        if self._return_key and self.key_col is not None:
            extra_args += self.key_buffer[idx],
        elif self._return_key and self.key_col is None:
            if self.worker_info is None:
                extra_args += self.idx_total,
            else :
                extra_args += f'{self.worker_info.id}-{self.idx_total}',
                
        return self.data_buffer[idx], self.targ_buffer[idx], *extra_args
        
    def __iter__(self):
        # Reset indexes used in __next__
        self.idx = 0
        self.idx_buffer = 0
        self.idx_file = 0
        self.curr_len = 0
        
        self.worker_info = get_worker_info()
        if self.worker_info is None:
            # Assign all files to workers
            self.h5workerfiles = self.h5files
            return self
        else:
            self.h5workerfiles = [
                _ for _iter, _ in enumerate(self.h5files)
                    if _iter % self.worker_info.num_workers == self.worker_info.id
            ]
            return self
        