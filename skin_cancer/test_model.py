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
            print(f'-- Using training configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate training configuration file in specified location; got {path}')

        path = os.path.join(args.test_dir, 'config', 'model.json')
        if os.path.exists(path):
            args.model_config = path
            print(f'-- Using model configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate model configuration file in specified location; got {path}')

    # Handle legacy -t/-m mode
    else:
        from build_model import get_model_config_path, get_train_config_path

        if args.model_config is None :
            raise ValueError(f'Cannot locate model configuration file.')

        else:
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
             
    # import set_environment
            
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    
    from kan_utils.utils import load_model, load_dict, save_dict, set_seed
    from kan_utils.dataset import smart_split_dataset
    from kan_utils.config import *
    from kan_utils.training import evaluate
    
    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset, get_groups
    from extract_statistics import extract_statistics
    # import custom_callbacks
    from build_model import build_model, get_training_subdir
    from custom_dataset import SkinCancerDataset, get_basic_transforms
    import custom_model

    device = torch.device(
        # 'cpu'
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
    model_config = load_config(args.model_config, locals=get_locals(custom_model))
    set_seed(train_config['seed'])
    training_subdir = get_training_subdir(
        training_stage  = 1,
        model_hash      = model_config['hash'], 
        train_hash      = train_config['hash'],
        top_dir         = args.test_dir,
        test_version    = args.test_version,
    )

    # Instantiate models
    img_enc = instantiate(model_config,'Image Encoder')
    heads   = {
        head : instantiate(model_config, head)
            for head in model_config['heads']
    }
    model   = build_model(
        img_enc = img_enc,
        **heads,
    )

    # Load model state dict
    fname = os.path.join(training_subdir, 'models', args.epoch)
    model = load_model(model, fname)

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

    df = normalize_dataset(expand_df_labels(build_dataset()))
    *_, test_loader = smart_split_dataset(
        splits          = train_config['splits'],
        full_dataset    = SkinCancerDataset(
            df,
            input_cols      = model_config['input'],
            output_cols     = model_config['output'],
            input_img_dims  = model_config['input_img_dim'],
            path_cols       = model_config['path_cols'],
            extra_transforms= get_basic_transforms(0.2, *model_config['input_img_dim'][-2:]),
            flatten         = model_config['flatten'] if 'flatten' in model_config.keys() else False,
            return_key      = True,
            return_type     ='dict',
        ),
        groups          = get_groups(),
        seed            = train_config['seed']
    )
    test_loader = DataLoader(
        test_loader, 
        shuffle         = False,
        batch_size      = train_config['batch_size'],
        num_workers     = os.cpu_count(),
        pin_memory      = device == torch.device('cuda'),
    )

    print('-- Using dataset split :', train_config['splits'])
    print('  -- Test       :', len(test_loader.dataset))

    test_metrics = evaluate(
        model,
        eval_dataloader   = test_loader,
        criteria          = eval_criteria,
        keep_copy         = True,
        checkpoint_path   = fname.replace('models','rslt'),
        epoch             = args.epoch,
        sample_weight     = train_config['sample_weight'],
        show_pbar         = True,
        device            = device,
    )
    hist_path = os.path.join(training_subdir,'history')
    history = load_dict(hist_path)

    history['test'] = {
        args.epoch : test_metrics
    }
    save_dict(history, hist_path)

    # Separate ground truth and predicted values
    rslt_path = os.path.join(training_subdir,'rslt')
    from tqdm import tqdm
    
    pbar = tqdm(pd.read_csv(os.path.join(rslt_path,f'{args.epoch}.csv'), index_col='Index', chunksize=10), dynamic_ncols=True)
    for _iter, test_df in enumerate(pbar):
        for head_name in heads.keys():
            gt_df : pd.DataFrame = test_df[[_ for _ in test_df.columns if f'targ_{head_name}' in _]]
            pr_df : pd.DataFrame = test_df[[_ for _ in test_df.columns if f'pred_{head_name}' in _]]

            if _iter == 0:
                os.makedirs(os.path.join(rslt_path, head_name), exist_ok=True)
                # print(gt_df.columns, model_config['output'][head_name])
                
            if sum([
                test_val in vals for vals in model_config['path_cols'].values()
                    for test_val in model_config['output'][head_name]
            ]) > 0:
                gt_df.columns = [_.lstrip(f'targ_{head_name}_') for _ in gt_df.columns]
                pr_df.columns = [_.lstrip(f'pred_{head_name}_') for _ in pr_df.columns]
                
                gt_df = pd.DataFrame(
                    data    =(pr_df[gt_df.columns].values * 255).round().astype('uint8'),
                    columns = pr_df.columns,
                    index   = pr_df.index 
                )
                pr_df = pd.DataFrame(
                    data    =(gt_df[gt_df.columns].values * 255).round().astype('uint8'),
                    columns = gt_df.columns,
                    index   = gt_df.index 
                )
            else :
                gt_df.columns = model_config['output'][head_name]
                pr_df.columns = model_config['output'][head_name]

                gt_df = normalize_dataset(gt_df, reverse=True)
                pr_df = normalize_dataset(pr_df, reverse=True)

            if _iter == 0:
                gt_df.to_csv(os.path.join(rslt_path, head_name, 'ground_truth.csv'))
                pr_df.to_csv(os.path.join(rslt_path, head_name,f'{args.epoch}.csv'))
            else :
                gt_df.to_csv(os.path.join(rslt_path, head_name, 'ground_truth.csv'), mode='a', header=False)
                pr_df.to_csv(os.path.join(rslt_path, head_name,f'{args.epoch}.csv'), mode='a', header=False)
        
        if _iter == 0:
            test_df.to_csv(os.path.join(rslt_path, f'rslt_{args.epoch}.csv'))
        else :
            test_df.to_csv(os.path.join(rslt_path, f'rslt_{args.epoch}.csv'), mode='a', header=False)

        # extract_statistics(pr_df, output_dir=rslt_path)
    os.remove(os.path.join(rslt_path,f'{args.epoch}.csv'))