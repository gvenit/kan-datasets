from typing import Union, Callable
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

def evaluate(
    model : Module,
    eval_dataloader : DataLoader,
    criteria : dict[str:Callable[[torch.Tensor],torch.Tensor]],
    keep_copy = True,
    checkpoint_path = None,
    epoch = None,
    show_pbar = True,
    device = torch.device('cpu'),
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
        for data in pbar:
            if len(data) == 3:
                data, target, key = data[0].to(device), data[1].to(device), data[2]
                # print(key)
                if isinstance(key, torch.Tensor):
                    key = key.tolist()
            else :
                data, target, key = data[0].to(device), data[1].to(device), None
            
            prediction = model(data)
            
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
            
        metrics = {
            name : float(criterion(prediction, target).cpu())
                for name, criterion in criteria.items()
        }
        
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
        
    return metrics