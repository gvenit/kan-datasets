#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

parser = ArgumentParser(
    description='Training script for the Ship Performance Clusterring Dataset.'
)

parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
parser.add_argument('--seed', dest='seed', type=int, default=42)
parser.add_argument('--patience', dest='patience', default=100)
parser.add_argument('--epochs', dest='epochs', default=500)
parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
parser.add_argument('--export', action='store_true',dest='export', help="Save the training configuration")
parser.add_argument('--test-version', dest='test_version', type=str, default=None)

args = parser.parse_args()

import torch
import torchmetrics
import hashlib

from kan_utils.config import *
from kan_utils.metrics import *
from prepare_dataset import build_dataset, expand_df_labels

df = expand_df_labels(build_dataset())

train_config = get_default_training_config()
train_config.update(
    object_to_config(
        torch.nn.HuberLoss,
        target_name     = 'criterion',
        reduction       = 'mean',
))
train_config['patience'] = args.patience
train_config['epochs'] = args.epochs
train_config.update(
    object_to_config(
        getattr(torch.optim, args.optimizer),
        target_name     = 'optimizer',
        weight_decay    = args.weight_decay,
        **({
            'momentum' : args.momentum
        } if args.optimizer in ('SGD', 'RMSprop') else {})
))
train_config['lr'] = args.lr
train_config['seed'] = args.seed
train_config['batch_size'] = args.batch_size
train_config['eval_criteria'] = {
    **object_to_config(
        torchmetrics.image.PeakSignalNoiseRatio,
        target_name     = 'PSNR',
        data_range      = 1.0,
        reduction       = 'elementwise_mean',
    ),
    **object_to_config(
        torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure,
        target_name     = 'MS-SSIM',
        data_range      = 1.0,
        reduction       = 'elementwise_mean',
    ),
}
def build_test_dir(train_config, top_dir = None, test_version = None,):
    pdir = os.path.join(
        find_class_name(train_config['scheduler']),
        find_class_name(train_config['optimizer']),
        '_'.join([
            'lr',
            str(train_config['lr'])
        ]),
        find_class_name(train_config['criterion']),
        '_'.join(['seed', str(train_config['seed'])]),
        '_'.join(['patience', str(train_config['patience'])]),
        '_'.join(['batch_size', str(train_config['batch_size'])]),
        '_'.join(['epochs', str(train_config['epochs'])]),
    )
    hashed = hashlib.sha256(pdir.encode()).hexdigest()
    pdir = os.path.join('train_config', 'img_enc_dec', hashed)
    if top_dir is not None:
        pdir = os.path.join(top_dir,pdir)
    if test_version is not None:
        pdir = os.path.join(pdir,'_'.join(['test',test_version]))
    return pdir, hashed

pdir, train_config['hash'] = build_test_dir(train_config, top_dir=args.dest_top_dir, test_version = args.test_version)
    
if args.hash:
    print(train_config['hash'])

if args.export :
    pdir = save_config(train_config, pdir)
    
if not args.hash:
    print(pdir)
