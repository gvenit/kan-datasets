from typing import Union, Callable, Literal, Any
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler 
import numpy as np
import os

from .evaluate import evaluate, get_callable_basis
from ..utils import save_model, load_model, save_dict

def train(
    model : Module,
    train_dataloader : DataLoader,
    eval_dataloader : DataLoader,
    criterion : Callable[[torch.Tensor],torch.Tensor],
    eval_criteria : dict[str:Callable[[torch.Tensor],torch.Tensor]],
    optimizer : Optimizer,
    scheduler : LRScheduler,
    epochs : int,
    patience : int = None,
    history : dict[str,dict[int,dict[str,Union[float,list[float]]]]] = {},
    start_epoch = 0,
    top_dirname = './train',
    device = torch.device('cpu'),
    evaluate_training = False,
    saving_steps : int = 1,
    show_pbar : Literal[None, 'external','internal'] = 'external',
    callbacks = get_callable_basis(),
    callbacks_arguments : dict[str, Any] = {},
) -> dict[str,dict[int,dict[str,Union[float,list[float]]]]]:
    '''
    A function to train a PyTorch model.
    
    :param model: The model to be trained.
    :type model: Module
    :param train_dataloader: The dataloader for training data.
    :type train_dataloader: DataLoader
    :param eval_dataloader: The dataloader for evaluation data.
    :type eval_dataloader: DataLoader
    :param criterion: The loss function to be used.
    :type criterion: Callable[[torch.Tensor], torch.Tensor]
    :param eval_criteria: The evaluation criteria to be used.
    :type eval_criteria: dict[str: Callable[[torch.Tensor], torch.Tensor]]
    :param optimizer: The optimizer to be used.
    :type optimizer: Optimizer
    :param scheduler: The learning rate scheduler to be used.
    :type scheduler: LRScheduler
    :param epochs: The number of epochs to train for.
    :type epochs: int
    :param patience: The number of epochs to wait before early stopping.
    :type patience: int
    :param history: The training history prior to this training session.
    :type history: dict[str, dict[int, dict[str, Union[float, list[float]]]]]
    :param start_epoch: The starting epoch number.
    :type start_epoch: int
    :param top_dirname: The top directory to save training results.
    :type top_dirname: str
    :param device: The device to be used for training.
    :type device: torch.device
    :param evaluate_training: Whether to evaluate on training data after each epoch.
    :type evaluate_training: bool
    :param saving_steps: The interval (in epochs) to save the model.
    :type saving_steps: int
    :param show_pbar: Whether to show progress bars. `external` shows an outer progress bar for epochs, `internal` shows an inner progress bar for iterations.
    :type show_pbar: Literal[None, 'external', 'internal']
    :param callbacks: Callback functions to be called at various stages of training.
    :type callbacks: dict[str, list[Callable[..., None]]]
    :param callbacks_arguments: Additional arguments to be passed to the callback functions.
    :type callbacks_arguments: dict[str, Any]
    :return: The training history including the new training session.
    :rtype: dict[str, dict[int, dict[str, float | list[float]]]]
    '''
    best_loss = float('inf')
    val_loss = float('inf')
    if len(history) == 0:
        history = {'train':{}, 'val':{}}
        
    elif len(history['train']) > 0:
        start_epoch = max()
        best_loss = min([value['loss'] for value in history['train'].values()])
            
    model.to(device)
    
    try :
        optimizer.to(device)
    except:
        pass
    
    if 'loss' not in eval_criteria.keys():
        eval_criteria = {
            'loss' : criterion,
            **eval_criteria
        }
    
    patience_counter = 0
    
    os.makedirs(top_dirname, exist_ok=True)
    model_dirname = os.path.join(top_dirname, 'models')
    os.makedirs(model_dirname, exist_ok=True)
        
    if show_pbar == 'external':
        pbar_epoch = tqdm(range(start_epoch+1, start_epoch+epochs+1), dynamic_ncols=True)
    else:
        pbar_epoch = range(start_epoch+1, start_epoch+epochs+1)
        
    descr = 'Epoch {epoch} -- Tr Loss {tr_loss:.5f} -- Val Loss {val_loss:.5f} -- Best {best_loss:.5f}'
    
    try :
        for epoch in pbar_epoch:
            hist_train_epoch = {}
            tr_loss = 0.
            
            model.train()
            
            if show_pbar == 'internal':
                pbar_iter = tqdm(train_dataloader, dynamic_ncols=True)
            else:
                pbar_iter = train_dataloader
                
            loc_kwargs = {
                'model'            : model,
                'epoch'            : epoch,
                'epochs'           : epochs,
                'best_loss'        : best_loss, 
                'dataloader'       : train_dataloader, 
                'patience'         : patience, 
                'patience_counter' : patience_counter,
                'criterion'        : criterion,
                'optimizer'        : optimizer,
                'scheduler'        : scheduler,
                'device'           : device,
                'history'          : history,
            }
            loc_kwargs.update(callbacks_arguments)
            for callback in callbacks['epoch_start']:
                callback(**loc_kwargs)

            try :
                for _iter, (data, target) in enumerate(pbar_iter, start=1):
                    if isinstance(data, tuple):
                        data = (_.to(device) for _ in data)
                    else :
                        data = data.to(device)
                        
                    if isinstance(data, tuple):
                        target = (_.to(device) for _ in target)
                    else :
                        target = target.to(device)
                    
                    loc_kwargs = {
                        'model'            : model,
                        'iteration'        : _iter,
                        'epoch'            : epoch, 
                        'epochs'           : epochs,
                        'data'             : data,
                        'target'           : target,
                        'dataloader'       : train_dataloader, 
                        'device'           : device,
                        'history'          : history,
                    }
                    loc_kwargs.update(callbacks_arguments)
                    for callback in callbacks['train_iter_start']:
                        callback(**loc_kwargs)

                    prediction = model(data)
                    
                    loss : torch.Tensor = criterion(prediction,target)
                    
                    if np.isnan(loss.detach().cpu()) :
                        data = data.cpu()
                        target = target.cpu()
                        print(data, target, loss)
                        save_dict(history, os.path.join(model_dirname,'history'))
                        raise ValueError(f'Encountered NaN value at epoch {epoch}')
                    else :
                        tr_loss += loss.detach().cpu()
                        
                    # Reset grads
                    optimizer.zero_grad()
                    # Calculate new grads
                    loss.backward()
                    
                    # if _iter == 1:
                    #     for param in model.parameters():
                    #         print(param.grad)
                            
                    # Update weights
                    optimizer.step()
                    
                    loc_kwargs = {
                        'model'         : model,
                        'iteration'     : _iter,
                        'epoch'         : epoch, 
                        'loss'          : loss.detach().cpu(),
                        'dataloader'    : train_dataloader, 
                        'optimizer'     : optimizer,
                        'device'        : device,
                        'history'       : history,
                    }
                    loc_kwargs.update(callbacks_arguments)
                    for callback in callbacks['train_iter_end']:
                        callback(**loc_kwargs)
                        
                    if show_pbar == 'internal':
                        pbar_iter.set_description(descr.format(epoch=epoch, tr_loss=tr_loss/_iter, val_loss=val_loss, best_loss=best_loss))
                    elif show_pbar == 'external':
                        pbar_epoch.set_description(descr.format(epoch=epoch, tr_loss=tr_loss/_iter, val_loss=val_loss, best_loss=best_loss))
                   
                del data, target, prediction, loss
                    
                tr_loss = float(tr_loss / len(train_dataloader))
                hist_train_epoch.update({'loss' : tr_loss})
                
                history['train'].update({epoch : hist_train_epoch})
                
                loc_kwargs = {
                    'model'         : model,
                    'epoch'         : epoch, 
                    'epochs'        : epochs,
                    'tr_loss'       : tr_loss,
                    'best_loss'     : best_loss, 
                    'dataloader'    : train_dataloader, 
                    'optimizer'     : optimizer,
                    'scheduler'     : scheduler,
                    'device'        : device,
                    'history'       : history,
                }
                loc_kwargs.update(callbacks_arguments)
                for callback in callbacks['train_end']:
                    callback(**loc_kwargs)

                if evaluate_training:
                    history['train'][epoch].update(
                        evaluate(
                            model           = model, 
                            eval_dataloader = train_dataloader, 
                            eval_criteria   = eval_criteria, 
                            device          = device, 
                            show_pbar       = False,
                            callbacks       = callbacks,
                            callbacks_arguments = {
                                'epoch' : epoch,
                                **callbacks_arguments
                            }
                        )
                    )
                    
                val_metrics = evaluate(
                    model, 
                    eval_dataloader = eval_dataloader, 
                    criteria        = eval_criteria, 
                    device          = device, 
                    show_pbar       = False,
                    callbacks       = callbacks,
                    callbacks_arguments = {
                        'epoch'  : epoch,
                        'epochs' : epochs,
                        **callbacks_arguments
                    }
                )
                history['val'].update({
                    epoch : val_metrics
                })
                val_loss = val_metrics['loss']
            
                if show_pbar == 'internal':
                    pbar_iter.set_description(descr.format(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss, best_loss=best_loss))
                    pbar_iter.close()
                    
                elif show_pbar == 'external':
                    pbar_epoch.set_description(descr.format(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss, best_loss=best_loss))
                    
                if saving_steps > 0 and (epoch % saving_steps == 0):
                    save_model(model, os.path.join(model_dirname,'last'), device)
                    save_dict(history, os.path.join(top_dirname,'history'))
                
                if best_loss > val_loss:
                    best_loss = val_loss
                    save_model(model, os.path.join(model_dirname,'best'), device)   
                    patience_counter = 0
                elif patience is not None and patience > 1:
                    patience_counter += 1
                    
                    if patience_counter > patience:
                        break
                    
                loc_kwargs = {
                    'model'            : model,
                    'epoch'            : epoch, 
                    'epochs'           : epochs,
                    'tr_loss'          : tr_loss,
                    'val_loss'         : val_loss,
                    'best_loss'        : best_loss, 
                    'train_dataloader' : train_dataloader, 
                    'eval_dataloader'  : eval_dataloader, 
                    'optimizer'        : optimizer,
                    'scheduler'        : scheduler,
                    'device'           : device,
                    'history'          : history,
                }
                loc_kwargs.update(callbacks_arguments)
                for callback in callbacks['epoch_end']:
                    callback(**loc_kwargs)
                    
                scheduler.step(val_loss)
                
            except Exception as e:
                if show_pbar == 'internal':
                    pbar_iter.close()
                raise e
            
        if show_pbar == 'external':
            pbar_epoch.close()
        
    except Exception as e:
        if show_pbar == 'external':
            pbar_epoch.close()
            
        loc_kwargs = {
            'model'            : model,
            'epoch'            : epoch, 
            'epochs'           : epochs,
            'val_loss'         : val_loss,
            'best_loss'        : best_loss, 
            'train_dataloader' : train_dataloader, 
            'eval_dataloader'  : eval_dataloader, 
            'optimizer'        : optimizer,
            'scheduler'        : scheduler,
            'device'           : device,
            'history'          : history,
            'exception'        : e, 
        }
        loc_kwargs.update(callbacks_arguments)
        for callback in callbacks['exception_raised']:
            callback(**loc_kwargs)
        for callback in callbacks['training_finished']:
            callback(**loc_kwargs)
            
        save_model(model, os.path.join(model_dirname,'last'), device)
        save_dict(history, os.path.join(top_dirname,'history'))
        
        if 'data' in locals():
            del data
        if 'target' in locals():
            del target
        if 'prediction' in locals():
            del prediction
        if 'loss' in locals():
            del loss
        
        raise e
    
    loc_kwargs = {
        'model'            : model,
        'epoch'            : epoch, 
        'epochs'           : epochs,
        'tr_loss'          : tr_loss,
        'val_loss'         : val_loss,
        'best_loss'        : best_loss, 
        'train_dataloader' : train_dataloader, 
        'eval_dataloader'  : eval_dataloader, 
        'optimizer'        : optimizer,
        'scheduler'        : scheduler,
        'device'           : device,
        'history'          : history,
    }
    loc_kwargs.update(callbacks_arguments)
    for callback in callbacks['training_finished']:
        callback(**loc_kwargs)
        
    if saving_steps < 1 or (epoch % saving_steps != 0):
        save_model(model, os.path.join(model_dirname,'last'), device)
        save_dict(history, os.path.join(top_dirname,'history'))
        
    return history