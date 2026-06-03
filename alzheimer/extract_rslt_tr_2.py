#!/usr/bin/env python3
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)


if __name__ == '__main__' :
    parser = ArgumentParser(
        description="Result extraction script for the 1st stage of the training scheme of the PROCESSED MRI Scans for Alzheimer's Detection Dataset."
    )

    parser.add_argument('-t', '--train-config', dest='train_config', help='The hash of the training configuration file.')
    parser.add_argument('-m', '--model-config', dest='model_config', help='The hash of the model configuration file.')
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')

    args = parser.parse_args()

    from build_model import *

    args.test_version = '_'.join(['test',args.test_version])

    # Check argument validity
    if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)
        
    else :
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    if args.model_config is None :
        raise ValueError(f'Cannot locate model configuration file.')
        
    else:
        img_hash = args.model_config
        args.model_config = get_model_config_path(
            training_stage  = 2,
            model_hash      = args.model_config,
            top_dir         = args.test_dir,
            test_version    = args.test_version,
        )
        args.model_config = f'{args.model_config}.json'
            
        if not os.path.exists(args.model_config) :
            raise ValueError(f'Cannot locate model configuration file in specified location; got {args.model_config}')
        else :
            print(f'-- Using model configuration path "{args.model_config}"')
        
    if args.train_config is None :
        raise ValueError(f'Cannot locate training configuration file.')
        
    else:
        train_hash = args.train_config
        args.train_config = get_train_config_path(
            training_stage  = 2,
            train_hash      = args.train_config,
            top_dir         = args.test_dir,
            test_version    = args.test_version,
        )
        args.train_config = f'{args.train_config}.json'
        
        if not os.path.exists(args.train_config) and not os.path.isabs(args.train_config):
            args.train_config = os.path.join(args.test_dir, args.train_config)
            
        if not os.path.exists(args.train_config) :
            raise ValueError(f'Cannot locate training configuration file in specified location; got {args.train_config}')
        else :
            print(f'-- Using training configuration path "{args.train_config}"')
                
    # import set_environment

    import pandas as pd
    import numpy as np
    # from sklearn.metrics import confusion_matrix
    from kan_utils.utils import load_dict
    import kan_utils.plotter as plotter
    from build_model import get_training_subdir

    # from kan_utils.config import *
    # from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
    import matplotlib.pyplot as plt

    training_subdir = get_training_subdir(
        training_stage  = 2,
        model_hash      = img_hash, 
        train_hash      = train_hash,
        top_dir         = args.test_dir,
        test_version    = args.test_version,
    )
    rslt_path = os.path.join(
        training_subdir,
        'rslt',
        args.epoch,
    )

    # Read training history
    history = load_dict(os.path.join(training_subdir,'history'))

    # Extract result statistics
    plots_path = os.path.join(training_subdir,'plot')
    os.makedirs(plots_path, exist_ok=True)

    ## Training vs Validation Loss
    epochs   = np.asarray(list(history['train'].keys()), dtype=int)
    tr_loss  = [_['loss'] for _ in history['train'].values()]
    # [print(_) for _ in history['val'].values()]
    val_loss = [_['loss'] for _ in history['val'].values()]

    plt.plot(epochs, tr_loss, val_loss)

    plt.title('Training vs Validation Loss')
    plt.legend(['training','validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    save_path = os.path.join(plots_path, 'tr_vs_val.png')
    plt.savefig(save_path)
    plt.close('all')
    print(f"Training vs Validation diagram saved to: {save_path}")

    ## Compare Predicted vs Ground truth images
    gt_df = pd.read_csv(os.path.join(os.path.dirname(rslt_path), 'ground_truth.csv'), index_col='Index')
    pr_df = pd.read_csv(os.path.join(os.path.dirname(rslt_path),f'{args.epoch}.csv'), index_col='Index')

    img_len = len(gt_df.columns) ** (1/3)
    img_len = int(round(img_len))

    os.makedirs(os.path.join(plots_path,'img_comp'), exist_ok=True)
    
    for idx in gt_df.index:
        gt_img = gt_df.loc[idx].values.reshape(img_len, img_len, img_len)
        pr_img = pr_df.loc[idx].values.reshape(img_len, img_len, img_len)
        
        for slice_axis in ('x','y','z',):
            fig, axes = plotter.plot_img_comparison(
                gt_img,
                pr_img,
                slice_axis = slice_axis,
                n_slices   = 5,
                figsize    = (15,5),
            )
            save_path = os.path.join(plots_path,'img_comp', f'idx-{idx}_axis-{slice_axis}.png')
            plt.savefig(save_path)
            plt.close('all')
            print(f"Image comparison diagram saved to: {save_path}")