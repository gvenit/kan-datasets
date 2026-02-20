#!/usr/bin/env python3

if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Testing script for the Ship Performance Clusterring Dataset.'
    )

    parser = ArgumentParser(
        description='Training script for the Particle Physics Event Classification Dataset.'
    )

    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')
    parser.add_argument('--no-pbar', action='store_true', dest='no_pbar')

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
            
    # import set_environment
            
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    
    from kan_utils.utils import load_model, load_dict, save_dict, set_seed
    from kan_utils.config import *
    from kan_utils.training import evaluate
    
    from prepare_dataset import get_dataset_paths
    from custom_dataset import DistributedH5Dataset
    # import custom_callbacks

    device = torch.device(
        # 'cpu'
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
    model_config = load_config(args.model_config)
    set_seed(train_config['seed'])

    # Instantiate models
    model = instantiate(model_config,'model')

    # Load model state dict
    fname = os.path.join(args.test_dir, 'models', args.epoch)
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

    test_loader = get_dataset_paths('test')
    test_loader    = DistributedH5Dataset(
        test_loader,
        buffer_size             = 32*train_config['batch_size'],
        input_cols              = model_config['input'],
        output_cols             = model_config['output'],
        key_col                 = 'event_no',
        task                    = train_config['task'],
        remove_mass_pt_window   = model_config['remove_mass_pt_window'],
        return_weights          = train_config['sample_weight'],
        return_key              = True,
        preprocess_data         = None,
        preprocess_targ         = None,
    )
    test_loader = DataLoader(
        test_loader, 
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
        show_pbar         = not args.no_pbar,
        device            = device,
    )

    hist_path = os.path.join(args.test_dir,'history')
    history = load_dict(hist_path)

    history['test'] = {
        args.epoch : test_metrics
    }
    save_dict(history, hist_path)

    # Separate ground truth and predicted values
    rslt_path = os.path.join(args.test_dir,'rslt')
    test_df = pd.read_csv(os.path.join(rslt_path,f'{args.epoch}.csv'), index_col='Index')

    gt_df = test_df[[_ for _ in test_df.columns if 'targ' in _]]
    pr_df = test_df[[_ for _ in test_df.columns if 'pred' in _]]

    if len(gt_df.columns) == 1 : # OR train_config['task'] == 'multiclass'
        if len(pr_df.columns) == 1 and len(model_config['output']) == 2:
            gt_df.columns = [f'Label_Is_{model_config['output'][1]}']
            if  model_config['outputs_logits'] :
                pr_df = pd.DataFrame(
                    data = torch.sigmoid(torch.tensor(pr_df.values),-1).numpy(),
                    index = pr_df.index,
                    columns = [f'Label_Is_{model_config['output'][1]}']
                )
            else :
                pr_df.columns = [f'Label_Is_{model_config['output'][1]}']
            
        elif len(pr_df.columns) > 1 :
            gt_df.columns = ['Label']
            gt_df = gt_df['Label'].map({
                key : val for key, val in enumerate(model_config['output'])
            })
            if  model_config['outputs_logits'] :
                pr_df = pd.DataFrame(
                    data = torch.softmax(torch.tensor(pr_df.values),-1).numpy(),
                    index = pr_df.index,
                    columns = [f'Label_Is_{_}' for _ in model_config['output']]
                )
            else :
                pr_df.columns = [f'Label_Is_{_}' for _ in model_config['output']]
        else :
            gt_df.columns = model_config['output']
            pr_df.columns = model_config['output']
    else :
        gt_df.columns = model_config['output']
        if  model_config['outputs_logits'] :
            pr_df = pd.DataFrame(
                data = torch.sigmoid(torch.tensor(pr_df.values),-1).numpy(),
                index = pr_df.index,
                columns = model_config['output'] 
            )
        else :
            pr_df.columns = model_config['output']
        
    gt_df.to_csv(os.path.join(rslt_path, 'ground_truth.csv'))
    pr_df.to_csv(os.path.join(rslt_path, f'{args.epoch}.csv'))
    test_df.to_csv(os.path.join(rslt_path, 'rslt.csv'))
