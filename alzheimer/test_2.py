#!/usr/bin/env python3
if __name__ == '__main__' :

    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description="Testing script for the 1st stage of the training scheme of PROCESSED MRI Scans for Alzheimer's Detection Dataset."
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
    import torch
    from torch.utils.data import DataLoader

    from kan_utils.utils import load_dict, save_dict, set_seed
    from kan_utils.dataset import smart_split_indices
    from kan_utils.performance import get_summary
    from kan_utils.config import *
    from kan_utils.training import evaluate

    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset, get_groups
    from custom_dataset import AlzheimerDataset
    from extract_statistics import extract_statistics
    import custom_model

    device = torch.device(
        # 'cpu'
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
    model_config = load_config(args.model_config, locals=get_locals(custom_model))
    img_config   = load_config(
        get_model_config_path(
            training_stage  = 1,
            model_hash      = model_config['img_hash'],
            top_dir         = args.test_dir,
            test_version    = args.test_version,
        ),
        locals=get_locals(custom_model)
    )

    # Instantiate models
    img_enc = instantiate(img_config,'img_enc')
    img_dec = instantiate(img_config,'img_dec')
    spt_enc = instantiate(model_config,'spt_enc')
    spt_dec = instantiate(model_config,'spt_dec')
    model   = build_train_2_model(
        spt_enc = spt_enc,
        spt_dec = spt_dec,
        img_enc = img_enc,
        img_dec = img_dec,
    )
    
    # Load model state dict
    load_train_2_model(
        model,
        img_hash    = model_config['img_hash'],
        spt_hash    = model_config['hash'],
        train_hash  = train_config['hash'],
        epoch       = args.epoch,
        top_dir     = args.test_dir,
        test_version= args.test_version,
    )
    
    # Instantiate evaluation criteria
    eval_criteria = {
        **weak_instantiate_all(train_config['eval_criteria'])
    }
    if 'loss' not in eval_criteria.keys():
        eval_criteria.update({
            'loss' : instantiate(train_config,'criterion'),
        })
    print('-- Evaluation Criteria :')
    if len(eval_criteria):
        for key, val in eval_criteria.items():
            print('  --', key, ':', val)
    else :
        print('  No evaluation criteria.')
        
    full_dataset = AlzheimerDataset(
        normalize_dataset(expand_df_labels(build_dataset())), 
        input_cols      = model_config['input'],
        output_cols     = model_config['output'],
        input_img_dims  = img_config['input_img_dim'],
        output_img_dims = img_config['output_img_dim'],
        return_key      = True, 
        path_col        = 'Path',
        orientation     = 'y',
    )

    set_seed(train_config['seed'])
    *_, test_loader = smart_split_indices(
        splits          = train_config['splits'],
        full_dataset    = full_dataset,
        groups          = get_groups(),
        seed            = train_config['seed']
    )
    training_subdir = get_training_subdir(
        training_stage  = 2,
        model_hash      = model_config['hash'], 
        train_hash      = train_config['hash'],
        top_dir         = args.test_dir,
        test_version    = args.test_version,
    )
    rslt_path = os.path.join(
        training_subdir,
        'rslt',
        f'{args.epoch}.csv'
    )
    test_loader = DataLoader(
        test_loader, 
        shuffle         = False,
        batch_size      = 4*train_config['batch_size'],
        num_workers     = os.cpu_count(),
        pin_memory      = False,
    )

    print('-- Using dataset split :', train_config['splits'])
    print('  -- Test       :', len(test_loader.dataset))

    # Instantiate callbacks
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

    test_metrics = evaluate(
        model,
        eval_dataloader     = test_loader,
        criteria            = eval_criteria,
        keep_copy           = True,
        checkpoint_path     = rslt_path,
        epoch               = args.epoch,
        show_pbar           = True,
        device              = device,
        callbacks           = callbacks,
        callbacks_arguments = callbacks_arguments,
    )

    hist_path = os.path.join(training_subdir,'history')
    history = load_dict(hist_path)

    history['test'] = {
        args.epoch : test_metrics
    }
    save_dict(history, hist_path)

    # Separate ground truth and predicted values
    test_df = pd.read_csv(rslt_path, index_col='Index')
    # test_df = pd.read_csv(rslt_path, index_col='Index', encoding='latin1')
    # test_df.index = pd.Index(full_dataset.get_keys(test_df.index.to_list()))
    test_df.to_csv(os.path.join(os.path.dirname(rslt_path),f'rslt_{args.epoch}.csv'))

    gt_df = test_df[[_ for _ in test_df.columns if 'targ' in _]]
    pr_df = test_df[[_ for _ in test_df.columns if 'pred' in _]]

    gt_df.columns = range(len(gt_df.columns))
    pr_df.columns = range(len(gt_df.columns))

    # gt_df = normalize_dataset(gt_df, reverse=True)
    # pr_df = normalize_dataset(pr_df, reverse=True)

    gt_df.to_csv(os.path.join(os.path.dirname(rslt_path), 'ground_truth.csv'))
    pr_df.to_csv(rslt_path)

    # extract_statistics(pr_df, output_dir=rslt_path)