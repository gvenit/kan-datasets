#!/usr/bin/env python3

if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Testing script for the MNIST Dataset.'
    )

    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')
    parser.add_argument('--no-pbar', action='store_true', dest='no_pbar')

    args = parser.parse_args()
    args.test_version = '_'.join(['test', args.test_version])

    # Check argument validity
    if os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)
    else:
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    args.test_dir = os.path.join(args.test_dir, args.hash, args.test_version)

    if args.hash is None:
        raise ValueError(f'Cannot locate training configuration file.')
    else:
        path = os.path.join(args.test_dir, 'config', 'train.json')
        if os.path.exists(path):
            args.train_config = path
            print(f'-- Using training configuration path "{path}"')
        else:
            raise ValueError(f'Cannot locate training configuration file.')

        path = os.path.join(args.test_dir, 'config', 'model.json')
        if os.path.exists(path):
            args.model_config = path
            print(f'-- Using model configuration path "{path}"')
        else:
            raise ValueError(f'Cannot locate model configuration file.')

    import torch
    from torch.utils.data import DataLoader
    import pandas as pd
    import albumentations as A

    from kan_utils.utils import load_model, load_dict, save_dict, set_seed
    from kan_utils.config import *
    from kan_utils.training import evaluate

    from prepare_dataset import build_dataset, get_dataset
    from custom_dataset import MNISTDataset
    import custom_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals(custom_model))
    set_seed(train_config['seed'])

    # Instantiate model and load weights
    model = instantiate(model_config, 'model')
    fname = os.path.join(args.test_dir, 'models', args.epoch)
    model = load_model(model, fname)

    # Instantiate callbacks
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

    # Instantiate evaluation criteria
    eval_criteria = {**weak_instantiate_all(train_config['eval_criteria'])}
    if 'loss' not in eval_criteria:
        eval_criteria['loss'] = instantiate(train_config, 'criterion')
    print('-- Evaluation Criteria :')
    for key, val in eval_criteria.items():
        print('  --', key, ':', val)

    # Build test dataset
    build_dataset()
    data, labels = get_dataset('test')

    preprocess_data = A.Compose([
            *([] if 'resize' not in train_config.keys() else [A.Resize(*train_config['resize'])]),
            A.Normalize(normalization = 'min_max_per_channel'), 
            # A.Normalize(normalization = 'standard'), #ImageNet mean/std
            A.ToTensorV2(),    
        ], 
        telemetry   = False,
        seed        = train_config['seed'],
    )
    
    test_loader = MNISTDataset(
        data, labels,
        task            = train_config['task'],
        return_weights  = train_config['sample_weight'],
        return_key      = True,
        preprocess_data = lambda **kwargs: preprocess_data(**kwargs)[list(kwargs.keys())[0]],
        flatten         = model_config['flatten'],
    )
    test_loader = DataLoader(
        test_loader,
        batch_size  = train_config['batch_size'],
        num_workers = os.cpu_count(),
        pin_memory  = device == torch.device('cuda'),
    )

    print(f'  -- Test : {len(test_loader.dataset)}')

    test_metrics = evaluate(
        model,
        eval_dataloader     = test_loader,
        criteria            = eval_criteria,
        keep_copy           = True,
        checkpoint_path     = fname.replace('models', 'rslt'),
        epoch               = args.epoch,
        sample_weight       = train_config['sample_weight'],
        show_pbar           = not args.no_pbar,
        device              = device,
        callbacks           = callbacks,
        callbacks_arguments = {
            'epoch': args.epoch,
            **callbacks_arguments,
        },
    )

    hist_path = os.path.join(args.test_dir, 'history')
    history = load_dict(hist_path)
    history['test'] = {args.epoch: test_metrics}
    save_dict(history, hist_path)

    # Separate ground truth and predicted values
    rslt_path = os.path.join(args.test_dir, 'rslt')
    test_df = pd.read_csv(os.path.join(rslt_path, f'{args.epoch}.csv'), index_col='Index')

    gt_df = test_df[[c for c in test_df.columns if 'targ' in c]]
    pr_df = test_df[[c for c in test_df.columns if 'pred' in c]]

    # Multiclass: single target column (class index), multiple prediction columns (logits/probs)
    gt_df.columns = ['Label']
    gt_df = gt_df['Label'].astype(int)

    if model_config['outputs_logits']:
        pr_df = pd.DataFrame(
            data    = torch.softmax(torch.tensor(pr_df.values.astype('float32')), dim=-1).numpy(),
            index   = pr_df.index,
            columns = [f'Label_Is_{c}' for c in model_config['output']],
        )
    else:
        pr_df.columns = [f'Label_Is_{c}' for c in model_config['output']]

    gt_df.to_csv(os.path.join(rslt_path, 'ground_truth.csv'))
    pr_df.to_csv(os.path.join(rslt_path, f'{args.epoch}.csv'))
    test_df.to_csv(os.path.join(rslt_path, 'rslt.csv'))
