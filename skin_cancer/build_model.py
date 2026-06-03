from collections import OrderedDict
import torch
import sys, os
from typing import overload
import hashlib

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

from kan_utils.utils import load_model, save_model, uses_momentum
from kan_utils.config import object_to_config, find_class_name
from kan_utils.models import MultiHead

_tr_type = {
    1 : 'enc_dec',
    # 2 : 'spt_enc_dec',
}

def build_training_dir(
    train_config, 
    top_dir = None, 
    test_version = None, 
    training_stage : int = 1
):
    pdir = os.path.join(*[
            '_'.join([
                find_class_name(train_config[obj]),
            ] + list([
                    str(_) for _ in train_config[f'{obj}_args']
                ] if f'{obj}_args' in train_config.keys() else []
            ) + list([
                    '_'.join([key, str(val)]) for key, val in train_config[f'{obj}_kwargs'].items()
                ] if f'{obj}_kwargs' in train_config.keys() else []
            ))
                for obj in ['scheduler', 'optimizer', 'criterion']
        ],
        '_'.join(['seed', str(train_config['seed'])]),
        '_'.join(['patience', str(train_config['patience'])]),
        '_'.join(['batch_size', str(train_config['batch_size'])]),
        '_'.join(['epochs', str(train_config['epochs'])]),
    )
    if training_stage > 1:
        pdir = os.path.join(train_config[f'tr{training_stage-1}_hash'], pdir)
        
    hashed = hashlib.sha1(pdir.encode()).hexdigest()
    pdir = os.path.join('train_config', hashed)
    # pdir = os.path.join('train_config', _tr_type[training_stage], hashed)
    if top_dir is not None:
        pdir = os.path.join(top_dir,pdir)
    if test_version is not None:
        pdir = os.path.join(pdir,test_version)
    
    return pdir, hashed 

def build_model_dir(
    model, 
    top_dir = None, 
    test_version = None,
):
    hashed = hashlib.sha1(repr(model).encode()).hexdigest()
    pdir = os.path.join('model_config', hashed)
    if top_dir is not None:
        pdir = os.path.join(top_dir,pdir)
    if test_version is not None:
        pdir = os.path.join(pdir,test_version)
    return pdir, hashed

def build_model(
    img_enc,
    **heads,
) :
    # if len(heads) > 1:
    heads = {
        'heads' : MultiHead(
        heads,
        return_type='dict'
    )}
    return torch.nn.Sequential(
        OrderedDict(
            img_enc = img_enc, 
            **heads,
        )
    )
    
def save_train_1_model(
    model,
    img_hash,
    train_hash,
    epoch,
    top_dir,
    test_version = 'test_0',
    device = torch.device('cpu'),
):
    img_enc_pth = os.path.join(top_dir, 'img_enc', img_hash, train_hash, test_version, epoch)
    img_dec_pth = os.path.join(top_dir, 'img_dec', img_hash, train_hash, test_version, epoch)
    
    os.makedirs(os.path.dirname(img_enc_pth), exist_ok=True)
    os.makedirs(os.path.dirname(img_dec_pth), exist_ok=True)
    
    model.to('cpu')
    save_model(model._modules['img_enc'], img_enc_pth)
    save_model(model._modules['img_dec'], img_dec_pth)
    
    model.to(device)
    
def load_train_1_model(
    model,
    img_hash,
    train_hash,
    epoch,
    top_dir,
    test_version = 'test_0',
    device = torch.device('cpu'),
):
    img_enc_pth = os.path.join(top_dir, 'img_enc', img_hash, train_hash, test_version, epoch)
    img_dec_pth = os.path.join(top_dir, 'img_dec', img_hash, train_hash, test_version, epoch)
    model.to('cpu')
    load_model(model._modules['img_enc'], img_enc_pth)
    load_model(model._modules['img_dec'], img_dec_pth)
    return model.to(device)

@overload
def build_train_2_model(
    spt_enc,
    spt_dec,
) :
    ...
    
@overload
def build_train_2_model(
    spt_enc,
    spt_dec,
    img_enc,
    img_dec,
) :
    ...
    
def build_train_2_model(
    spt_enc,
    spt_dec,
    img_enc = None,
    img_dec = None,
) :
    if img_enc is None and img_dec is None :
        return torch.nn.Sequential(
            OrderedDict([
                ('spt_enc', spt_enc), 
                ('spt_dec', spt_dec),
            ])
        )
    return torch.nn.Sequential(
        OrderedDict([
            ('img_enc', img_enc), 
            ('spt_enc', spt_enc), 
            ('spt_dec', spt_dec),
            ('img_dec', img_dec),
        ])
    )
    
def save_train_2_model(
    model,
    spt_hash,
    train_hash,
    epoch,
    top_dir = None,
    test_version = 'test_0',
    device = torch.device('cpu'),
):
    model.to('cpu')
    for name, module in model._modules.items():
        pth = os.path.join(top_dir, name, spt_hash, train_hash, test_version, epoch)
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        save_model(module, pth)
        
    model.to(device)
    
def load_train_2_model(
    model,
    img_hash,
    spt_hash,
    train_hash,
    epoch,
    top_dir = None,
    test_version = 'test_0',
    device = torch.device('cpu'),
):
    model.to('cpu')
    for name, module in model._modules.items():
        if name.startswith('img'):
            pth = os.path.join(top_dir, name, img_hash, train_hash, test_version, epoch)
        else:
            pth = os.path.join(top_dir, name, spt_hash, train_hash, test_version, epoch)
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        save_model(module, pth)
        
    return model.to(device)

def get_training_subdir(
    training_stage : int,
    model_hash,
    train_hash,
    top_dir = None,
    test_version = 'test_0',
):
    pth = os.path.join(f'train_{training_stage}', model_hash, train_hash, test_version)
    if top_dir is not None:
        return os.path.join(top_dir, pth)

def get_train_config_path(
    training_stage : int,
    train_hash,
    top_dir = None,
    test_version = 'test_0',
):
    pth = os.path.join('train_config', train_hash, test_version)
    # pth = os.path.join('train_config', _tr_type[training_stage], train_hash, test_version)
    if top_dir is not None:
        pth = os.path.join(top_dir, pth)
        
    return pth

def get_model_config_path(
    training_stage : int,
    model_hash,
    top_dir = None,
    test_version = 'test_0',
):
    pth = os.path.join('model_config', model_hash, test_version)
    # pth = os.path.join('model_config', _tr_type[training_stage], model_hash, test_version)
    if top_dir is not None:
        pth = os.path.join(top_dir, pth)
        
    return pth

def housekeep(
    training_stage : int,
    model,
    model_hash,
    train_hash,
    top_dir = None,
    test_version = 'test_0',
    device = torch.device('cpu'),
) :
    tr_subdir = get_training_subdir(
        training_stage  = training_stage,
        model_hash      = model_hash, 
        train_hash      = train_hash,
        top_dir         = top_dir,
        test_version    = test_version,
    )
    if training_stage == 1 :
        # Split best model state dict to separate files
        pth = os.path.join(tr_subdir, 'models', '{epoch}')
        
        for epoch in os.listdir(os.path.dirname(pth)):
            try :
                epoch = os.path.splitext(epoch)[0]
                load_model(model, pth.format(epoch=epoch))
                _pth = save_model(model, pth.format(epoch=epoch))
                save_train_1_model(
                    model,
                    img_hash    = model_hash, 
                    train_hash  = train_hash,
                    epoch       = epoch,
                    top_dir     = top_dir,
                    test_version= test_version,
                    device      = device,
                )
                os.remove(_pth)
            except Exception as e:
                print(e)
                
        os.removedirs(os.path.dirname(pth))
    if training_stage == 2 :
        # Split best model state dict to separate files
        pth = os.path.join(tr_subdir, 'models', '{epoch}')
        
        for epoch in os.listdir(os.path.dirname(pth)):
            try :
                epoch = os.path.splitext(epoch)[0]
                load_model(model, pth.format(epoch=epoch))
                _pth = save_model(model, pth.format(epoch=epoch))
                save_train_2_model(
                    model,
                    spt_hash    = model_hash, 
                    train_hash  = train_hash,
                    epoch       = epoch,
                    top_dir     = top_dir,
                    test_version= test_version,
                    device      = device,
                )
                os.remove(_pth)
            except Exception as e:
                print(e)
                
        os.removedirs(os.path.dirname(pth))
        
def check_heads(model_config, args):
    for head in model_config['heads']:
        assert head in model_config['output'].keys()
        
    for head in model_config['output'].keys():
        if head not in model_config['heads']:
            model_config['output'].pop(head)
            continue
        if 'output_img_dim' not in model_config:
            model_config['output_img_dim'] = {}
            
        if 'Image' in model_config['output'][head]:
            model_config['output_img_dim'][head] = [args.channels, *args.resize]
    
        if 'Mask' in model_config['output'][head]:
            model_config['output_img_dim'][head] = [1, *args.resize]
