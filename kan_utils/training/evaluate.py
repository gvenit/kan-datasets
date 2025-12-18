from typing import Union, Callable, Literal, Any
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

def get_callable_basis() :
    '''Basis for callable categories of callable functions.
    All callables should be of form:
    
        def callable(arg1, ..., argN, **kwargs):
            ...

    
    Available callback stages
    -------------------------
        epoch_start: 
            At the start of an epoch (before training).
        train_iter_start: 
            At the start of an iteration (during training).
        train_iter_end: 
            At the end of an iteration (during training).
        train_end: 
            At the end of the training in an epoch (after training).
        eval_start: 
            At the start of an evaluation (before validation).
        eval_iter_start: 
            At the start of an iteration (during validation).
        eval_iter_end: 
            At the end of an iteration (during validation).
        eval_metrics_start: 
            At the end of model prediction, before calculating metrics (during validation)
        eval_end: 
            At the end of an evaluation (after validation).
        epoch_end: 
            At the end of an epoch (after scheduler).
        exception_raised:
            When an exception is raised. 
        training_finished:
            At the end of all training (after all epochs are executed or when the patience counter reaches the maximum value).
            
    Returns
    -------
        dict[str, (...) -> None]
    '''
    return {
        'epoch_start'           : [],
        'train_iter_start'      : [],
        'train_iter_end'        : [],
        'train_end'             : [],
        'eval_start'            : [],
        'eval_iter_start'       : [],
        'eval_iter_end'         : [],
        'eval_metrics_start'    : [],
        'eval_end'              : [],
        'epoch_end'             : [],
        'exception_raised'      : [], 
        'training_finished'     : [],
    }

def evaluate(
    model : Module,
    eval_dataloader : DataLoader,
    criteria : dict[str:Callable[[torch.Tensor],torch.Tensor]],
    keep_copy = True,
    checkpoint_path = None,
    epoch = None,
    show_pbar = True,
    device = torch.device('cpu'),
    callbacks = get_callable_basis(),
    callbacks_arguments : dict[str, Any] = {},
) -> dict[str, Union[float,list[float]]]:
    if len(criteria) == 0:
        return {}
    
    model.eval()
    model.to(device)
    
    preds = []
    targs = []
    keys  = []
    
    if show_pbar:
        pbar = tqdm(eval_dataloader)
        
    else :
        pbar = eval_dataloader

    with torch.no_grad():    
        loc_kwargs = {
            'model'            : model,
            'epoch'            : epoch, 
            'eval_dataloader'  : eval_dataloader, 
            'device'           : device,
        }
        loc_kwargs.update(callbacks_arguments)
        for callback in callbacks['eval_start']:
            callback(**loc_kwargs)
            
        for data, target, *key in pbar:
            if isinstance(data, tuple):
                data = (_.to(device) for _ in data)
            else :
                data = data.to(device)
                
            if isinstance(data, tuple):
                target = (_.to(device) for _ in target)
            else :
                target = target.to(device)
            
            if len(key) > 0:
                key = key[0]
                if isinstance(key, torch.Tensor):
                    key = key.tolist()
            else :
                key = None
                          
            loc_kwargs = {
                'model'         : model,
                'epoch'         : epoch, 
                'data'          : data,
                'target'        : target,
                'key'           : key,
                'dataloader'    : eval_dataloader, 
                'device'        : device,
            }
            loc_kwargs.update(callbacks_arguments)
            for callback in callbacks['eval_iter_start']:
                callback(**loc_kwargs)
                
            prediction = model(data)
            
            loc_kwargs = {
                'model'         : model,
                'epoch'         : epoch, 
                'prediction'    : prediction,
                'target'        : target,
                'key'           : key,
                'dataloader'    : eval_dataloader, 
                'device'        : device,
            }
            loc_kwargs.update(callbacks_arguments)
            for callback in callbacks['eval_iter_end']:
                callback(**loc_kwargs)
                
            preds.append(prediction.cpu())
            targs.append(target.cpu())
            if key is not None:
                keys.extend(key)
                # print(keys)
            
        prediction = torch.cat(preds)
        target = torch.cat(targs)
        
        try :
            prediction = prediction.to(device)
            target = target.to(device)
        except:
            prediction = prediction.cpu()
            target = target.cpu()
                
        if show_pbar:
            pbar.close()
            
        loc_kwargs = {
            'eval_criteria' : criteria,
            'epoch'         : epoch, 
            'prediction'    : prediction,
            'target'        : target,
            'key'           : keys,
            'dataloader'    : eval_dataloader, 
            'device'        : device,
        }
        loc_kwargs.update(callbacks_arguments)
        for callback in callbacks['eval_metrics_start']:
            callback(**loc_kwargs)
            
        for criterion in criteria.values():
            try :
                criterion.to(device)
            except :
                pass
            
        metrics = {}
        for name, criterion in criteria.items():
            try :
                metrics[name] = criterion(prediction, target).double().cpu().tolist()
            except :
                metrics[name] = criterion(prediction, target)
        
    if keep_copy and len(keys) > 0 and checkpoint_path is not None:
        rslt_path = os.path.join(os.path.dirname(checkpoint_path), "rslt.csv" if epoch is None else f"{epoch}.csv")
        
        os.makedirs(os.path.dirname(rslt_path), exist_ok=True)
        
        prediction = prediction.cpu()
        target = target.cpu()
        
        df  = pd.DataFrame({
            'Index' : keys,
            **{
            f"targ_{_iter}" : target[:,_iter] 
                for _iter in range(target.shape[-1])
            },
            **{
            f"pred_{_iter}" : prediction[:,_iter] 
                for _iter in range(prediction.shape[-1])
            },
        }).set_index('Index').sort_index()
        df.to_csv(rslt_path)
        
        print(f'Results written to "{rslt_path}"')
        del df
        
    loc_kwargs = {
        'metrics'          : metrics,
        'epoch'            : epoch, 
        'prediction'       : prediction,
        'target'           : target,
        'key'              : key,
        'eval_dataloader'  : eval_dataloader, 
        'device'           : device,
    }
    loc_kwargs.update(callbacks_arguments)
    for callback in callbacks['eval_end']:
        callback(**loc_kwargs)
        
    return metrics