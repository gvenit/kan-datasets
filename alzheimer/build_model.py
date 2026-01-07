from collections import OrderedDict
import torch
import sys, os

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

from kan_utils.utils import load_model, save_model

def build_train_1_model(
    img_enc,
    img_dec,
) :
    return torch.nn.Sequential(
        OrderedDict([
            ('img_enc', img_enc), 
            ('img_dec', img_dec),
        ])
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

def get_training_subdir(
    training_stage : int,
    img_hash,
    train_hash,
    top_dir,
    test_version = 'test_0',
):
    if training_stage == 1 :
        pth = os.path.join('train_1', img_hash, train_hash, test_version)
        if top_dir is not None:
            return os.path.join(top_dir, pth)
    else :
        raise NotImplementedError(f'Implemented only for stage 1: got {training_stage}')

def housekeep(
    training_stage : int,
    model,
    img_hash,
    train_hash,
    top_dir,
    test_version = 'test_0',
    device = torch.device('cpu'),
) :
    if training_stage == 1 :
        # Split best model state dict to separate files
        pth = os.path.join(top_dir, 'models', '{epoch}')
        
        for epoch in os.listdir(os.path.dirname(pth)):
            try :
                epoch = os.path.splitext(epoch)[0]
                load_model(model, pth.format(epoch=epoch))
                _pth = save_model(model, pth.format(epoch=epoch))
                save_train_1_model(
                    model,
                    img_hash    = img_hash, 
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

        # Move history directory
        if os.path.exists(os.path.join(top_dir, 'history.json')):
            os.renames(
                os.path.join(top_dir, 'history.json'),
                os.path.join(top_dir, 'train_1', img_hash, train_hash, test_version, 'history.json'),
            )
    
        