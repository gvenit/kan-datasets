import torch
from torch.nn import Module
import os, json
import numpy as np

def save_model(model: Module, fname:str, device = torch.device('cpu')):
    if os.path.splitext(fname)[-1] not in ('.pt','.pth'):
        fname = f'{fname}.pt'
    torch.save(
        model.state_dict(),
        fname
    )
    model.to(device)
    
def load_model(model: Module, fname:str, device = torch.device('cpu')):
    if os.path.splitext(fname)[-1] not in ('.pt','.pth'):
        fname = f'{fname}.pt'
    state_dict = torch.load(fname)
    model.cpu().load_state_dict(state_dict)
    return model.to(device)

def save_dict(obj : dict, fname : str):
    if os.path.splitext(fname)[-1] != '.json':
        fname = f'{fname}.json'
    with open(fname, 'w') as fwriter:
        json.dump(obj, fwriter, indent=2)
    return fname

def load_dict(fname) -> dict:
    if os.path.splitext(fname)[-1] != '.json':
        fname = f'{fname}.json'
    with open(fname, 'r') as freader:
        obj = json.load(freader)    
    return obj

def set_seed(seed):
    torch.manual_seed(seed),
    # torch.random.manual_seed(seed)
    np.random.seed(seed)
    
def expand_value(val, size):
    if not hasattr(val, '__iter__'):
        val = [val for _ in range(size)]

    if len(val) < size:
        val = val + [val[-1] for _ in range(size-len(val))]

    assert len(val) == size, f"Size missmatch; expected size {size}; got {len(val)} \n {val}"
    return val
