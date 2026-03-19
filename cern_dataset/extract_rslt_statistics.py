#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the Particle Physics Event Classification Dataset.'
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

    if args.hash is None :
            raise ValueError(f'Cannot locate training configuration file.')
    else :
        path = os.path.join(args.test_dir, 'config', 'train.json')
        if os.path.exists(path):
            args.train_config = path
            print(f'-- Using training configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate training configuration file.')
            
        path = os.path.join(args.test_dir, 'config', 'model.json')
        if os.path.exists(path):
            args.model_config = path
            print(f'-- Using model configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate model configuration file.')
            
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    from kan_utils.utils import load_dict
    import kan_utils.plotter as plotter

    from kan_utils.config import *
    # from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
    from prepare_dataset import create_labels

    # Check configuration file validity
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)

    # Read training history
    history = load_dict(os.path.join(args.test_dir, 'history'))
    test = history['test'][args.epoch]
    
    #Print basic statistics
    print(f'Loss for epoch "{args.epoch}": {test['loss']}')
    
    for key in ['Accuracy', 'F1Score', 'Precision', 'Recall', 'MSE', 'MAE', 'AUROC']:
        if key in test.keys():
            print(f'{key} for epoch "{args.epoch}": {test[key]}')

    # Extract result statistics
    plots_path = os.path.join(args.test_dir,'plot')
    os.makedirs(plots_path, exist_ok=True)

    # Training vs Validation Loss
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
    
    if 'PrecisionRecallCurve' in test.keys():
        import torch
        import torchmetrics
        from kan_utils.config import *
        
        # train_config = load_config(args.train_config, locals=get_locals())
        
        pr_curve = instantiate(train_config['eval_criteria'],'PrecisionRecallCurve')
        
        plt_args = [torch.tensor(_).float() for _ in test['PrecisionRecallCurve']]
        fig, ax = pr_curve.plot(curve = plt_args, score=True)
            
        save_path = os.path.join(plots_path, 'precision_recall_curve.png')
        plt.savefig(save_path)
        plt.close('all')
        print(f"Precision-Recall Curve saved to: {save_path}")
        
    # Read ground truth and predictied values
    rslt_path = os.path.join(args.test_dir,'rslt')

    gt_df = pd.read_csv(os.path.join(rslt_path, 'ground_truth.csv'), index_col='Index')
    pr_df = pd.read_csv(os.path.join(rslt_path, f'{args.epoch}.csv'), index_col='Index')

    # Read Categories
    categories = create_labels(
        features    = model_config['input'],
        labels      = model_config['output'],
        unify_labels= train_config['task'] in ('as_binary', 'multiclass'),
        save        = False,
    )
    os.makedirs(os.path.join(plots_path, args.epoch), exist_ok=True)

    # Extract Confusion Matrices for each set of categories
    categorical_cols = []
    for category, types in categories.items() :
        class_names = list(types.keys())
        
        # Find columns of the specified category
        gt_cols = [col for col in gt_df.columns if category in col]
        pr_cols = [col for col in pr_df.columns if category in col]
        categorical_cols.extend(gt_cols)
        categorical_cols.extend(pr_cols)
        
        # Get DataFrame slices
        gt_slice = gt_df[gt_cols].copy()
        pr_slice = pr_df[pr_cols].copy()
        
        if len(gt_cols) == 1:
            cat = {val : key for key, val in types.items()}
            gt_type = gt_slice[gt_cols[0]].apply(lambda row: cat[row > 0.5])
        else :
            # Fix DataFrames
            gt_slice.columns = [col[col.find('_Is_')+4:] for col in gt_slice.columns]
            
            # Find probabilities of the unknown class
            if train_config['task'] == 'multiclass' :
                gt_type = gt_slice.aggregate('argmax', axis=1).map(dict(enumerate(gt_slice.columns)))
            else :
                gt_type = gt_slice
                
        if len(pr_cols) == 1:
            cat = {val : key for key, val in types.items()}
            pr_type = pr_slice[pr_cols[0]].apply(lambda row: cat[row > 0.5])
        else :
            # Fix DataFrames
            pr_slice.columns = [col[col.find('_Is_')+4:] for col in pr_slice.columns]
            
            # Find probabilities of the unknown class
            if train_config['task'] == 'multiclass' :
                pr_type = pr_slice.apply(np.argmax, axis=1).map(dict(enumerate(pr_slice.columns)))
            else :
                pr_type = pr_slice
                
        cm = confusion_matrix(gt_type.values, pr_type.values, labels=class_names)
        save_path = os.path.join(plots_path, args.epoch, f'cm_{category}.png')
        plotter.plot_confusion_matrix(
            cm, 
            class_names, 
            normalize = True, 
            title     = f'Confusion Matrix : {category}',
            save_path = save_path
        )
        print(f"{category} Confusion matrix saved to: {save_path}")
        plt.close('all')

    # Plot Reggression-type Columns
    idx = gt_df.index.values
    for col in gt_df.columns[np.isin(gt_df.columns, categorical_cols, invert=True)]:
        plt.plot(idx, gt_df[col], pr_df[col])

        plt.title(col)
        plt.legend(['Ground Truth','Prediction'])
        plt.xlabel('Index')
        plt.ylabel(col)

        save_path = os.path.join(plots_path, args.epoch, f'{col}.png')
        plt.savefig(save_path)
        print(f"{col} Diagram saved to: {save_path}")
        plt.close('all')
