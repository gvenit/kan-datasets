#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the HAM10000 Skin Cancer Dataset.'
    )
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('-t', '--train-config', dest='train_config', help='The hash of the training configuration file.')
    parser.add_argument('-m', '--model-config', dest='model_config', help='The hash of the model configuration file.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the unified configuration.')
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')

    args = parser.parse_args()
    args.test_version = '_'.join(['test',args.test_version])

    # Check argument validity
    if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)

    else :
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    # Handle unified --hash mode (from create_configs_logits.py)
    if args.hash is not None:
        args.test_dir = os.path.join(args.test_dir, args.hash, args.test_version)

        path = os.path.join(args.test_dir, 'config', 'train.json')
        if os.path.exists(path):
            args.train_config = path
            args.train_hash = args.hash
            print(f'-- Using training configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate training configuration file in specified location; got {path}')

        path = os.path.join(args.test_dir, 'config', 'model.json')
        if os.path.exists(path):
            args.model_config = path
            args.model_hash = args.hash
            print(f'-- Using model configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate model configuration file in specified location; got {path}')

    # Handle legacy -t/-m mode
    else:
        from build_model import get_model_config_path, get_train_config_path

        if args.model_config is None :
            raise ValueError(f'Cannot locate model configuration file.')

        else:
            args.model_hash   = args.model_config
            args.model_config = get_model_config_path(
                training_stage  = 1,
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
            args.train_hash   = args.train_config
            args.train_config = get_train_config_path(
                training_stage  = 1,
                train_hash      = args.train_config,
                top_dir         = args.test_dir,
                test_version    = args.test_version,
            )
            args.train_config = f'{args.train_config}.json'

            if not os.path.exists(args.train_config) :
                raise ValueError(f'Cannot locate training configuration file in specified location; got {args.train_config}')
            else :
                print(f'-- Using training configuration path "{args.train_config}"')

    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    from kan_utils.utils import load_dict
    import kan_utils.plotter as plotter
    from kan_utils.config import *
    # from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
    
    import custom_model
    from build_model import get_training_subdir

    # Check configuration file validity
    # train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
    model_config = load_config(args.model_config, locals=get_locals(custom_model))
    training_subdir = get_training_subdir(
        training_stage  = 1,
        model_hash      = args.model_hash, 
        train_hash      = args.train_hash,
        top_dir         = args.test_dir,
        test_version    = args.test_version,
    )
    print(training_subdir, args.test_dir)

    # Read training history
    history = load_dict(os.path.join(training_subdir, 'history'))
    test = history['test'][args.epoch]
    
    #Print basic statistics
    print(f'Loss for epoch "{args.epoch}": {test['loss']}')
    
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
        import extract_statistics
        
        train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
        
        pr_curve = instantiate(train_config['eval_criteria'],'PrecisionRecallCurve')
        
        plt_args = [torch.tensor(_).float() for _ in test['PrecisionRecallCurve']]
        fig, ax = pr_curve.plot(curve = plt_args, score=True)
            
        save_path = os.path.join(plots_path, 'precision_recall_curve.png')
        plt.savefig(save_path)
        plt.close('all')
        print(f"Precision-Recall Curve saved to: {save_path}")
        
    # Read ground truth and predictied values
    rslt_path = os.path.join(training_subdir,'rslt')

    for head_name in os.listdir(rslt_path):
        if head_name in model_config['heads'] and os.path.isdir(os.path.join(rslt_path,head_name)):
            gt_df = pd.read_csv(os.path.join(rslt_path, head_name, 'ground_truth.csv'), index_col='Index')
            pr_df = pd.read_csv(os.path.join(rslt_path, head_name,f'{args.epoch}.csv'), index_col='Index')
            if head_name in model_config['output_img_dim'] :
                ## Compare Predicted vs Ground truth images
                img_len = len(gt_df.columns) ** (1/3)
                img_len = int(round(img_len))

                os.makedirs(os.path.join(plots_path, head_name,'img_comp'), exist_ok=True)
                
                for idx in gt_df.index:
                    gt_img = gt_df.loc[idx].values.reshape(img_len, img_len, img_len)
                    pr_img = pr_df.loc[idx].values.reshape(img_len, img_len, img_len)
                    
                    fig, axes = plotter.plot_img_comparison(
                        gt_img,
                        pr_img,
                        slice_axis = 'c',
                        n_slices   = 5,
                        figsize    = (15,5),
                        axis_seq   = 'chw',
                    )
                    save_path = os.path.join(plots_path,'img_comp', f'{idx}.png')
                    plt.savefig(save_path)
                    plt.close('all')
                    print(f"Image comparison diagram saved to: {save_path}")
            else :
                # Read Categories
                categories = load_dict(os.path.join(THIS_DIR, 'dataset', 'labels'))
                os.makedirs(os.path.join(plots_path, args.epoch), exist_ok=True)

                # Extract Confusion Matrices for each set of categories
                categorical_cols = []
                for category, types in categories.items() :
                    class_names = list(types.keys())
                    
                    # Find columns of the specified category
                    cols = [col for col in gt_df.columns if category in col]
                    categorical_cols.extend(cols)
                    
                    # Get DataFrame slices
                    gt_slice = gt_df[cols].copy()
                    pr_slice = pr_df[cols].copy()
                    
                    if len(cols) == 1:
                        cat = {val : key for key, val in types.items()}
                        gt_type = gt_slice[category].apply(lambda row: cat[row > 0.5])
                        pr_type = pr_slice[category].apply(lambda row: cat[row > 0.5])
                    else :
                        # Fix DataFrames
                        gt_slice.columns = [col[col.find('_Is_')+4:] for col in gt_slice.columns]
                        pr_slice.columns = [col[col.find('_Is_')+4:] for col in pr_slice.columns]
                        
                        # Get probabilities with Softmax
                        pr_slice = pr_slice.loc[pr_slice.index].apply(np.exp)
                        pr_slice = pr_slice.loc[pr_slice.index].apply(
                            (lambda row : row / row.sum()),
                            axis=1
                        )
                        
                        # Find probabilities of the unknown class
                        gt_type = gt_slice.apply(
                            lambda row: gt_slice.columns[np.argmax(row)],
                            axis=1
                        )
                        pr_type = pr_slice.apply(
                            lambda row: pr_slice.columns[np.argmax(row)],
                            axis=1
                        )
                        
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
