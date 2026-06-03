#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Extract statistics for Skin Cancer (Simplified).'
    )
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')

    args = parser.parse_args()
    args.test_version = '_'.join(['test',args.test_version])

    # Check argument validity
    if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)
    else :
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    args.test_dir = os.path.join(args.test_dir, args.hash, args.test_version)

    path = os.path.join(args.test_dir, 'config', 'train.json')
    if os.path.exists(path):
        args.train_config = path
        print(f'-- Using training configuration path "{path}"')
    else :
        raise ValueError(f'Cannot locate training configuration file in specified location; got {path}')

    path = os.path.join(args.test_dir, 'config', 'model.json')
    if os.path.exists(path):
        args.model_config = path
        print(f'-- Using model configuration path "{path}"')
    else :
        raise ValueError(f'Cannot locate model configuration file in specified location; got {path}')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from kan_utils.utils import load_dict
    from kan_utils.config import *

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals())
    training_subdir = args.test_dir
    print(f"Training subdirectory: {training_subdir}")

    # Read training history
    history = load_dict(os.path.join(training_subdir, 'history'))
    test = history['test'][args.epoch]

    # Print basic statistics
    print(f'Loss for epoch "{args.epoch}": {test["loss"]}')

    for key_type in ['Accuracy', 'F1Score', 'Precision', 'Recall', 'MSE', 'MAE', 'AUROC']:
        for key in test.keys():
            if key.startswith(key_type):
                print(f'{key} for epoch "{args.epoch}": {test[key]}')

    # Extract result statistics
    plots_path = os.path.join(training_subdir,'plot')
    os.makedirs(plots_path, exist_ok=True)

    ## Training vs Validation Loss
    epochs   = np.asarray(list(history['train'].keys()), dtype=int)
    tr_loss  = [_['loss'] for _ in history['train'].values()]
    val_loss = [_['loss'] for _ in history['val'].values()]

    plt.figure()
    plt.plot(epochs, tr_loss, label='training')
    plt.plot(epochs, val_loss, label='validation')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    save_path = os.path.join(plots_path, 'tr_vs_val.png')
    plt.savefig(save_path)
    plt.close('all')
    print(f"Training vs Validation diagram saved to: {save_path}")

    # If we have accuracy metrics, plot them too
    if 'Accuracy' in history['train'].get(list(history['train'].keys())[0], {}):
        tr_acc  = [_['Accuracy'] for _ in history['train'].values()]
        val_acc = [_['Accuracy'] for _ in history['val'].values()]

        plt.figure()
        plt.plot(epochs, tr_acc, label='training')
        plt.plot(epochs, val_acc, label='validation')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        save_path = os.path.join(plots_path, 'tr_vs_val_acc.png')
        plt.savefig(save_path)
        plt.close('all')
        print(f"Training vs Validation Accuracy diagram saved to: {save_path}")

