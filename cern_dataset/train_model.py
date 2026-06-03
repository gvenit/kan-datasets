#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the Particle Physics Event Classification Dataset.'
    )

    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
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
    
    import torch
    from torch.utils.data import DataLoader

    from kan_utils.config import *
    from kan_utils.training import train
    from kan_utils.utils import set_seed, apply_to_tensor
    from kan_utils.performance import get_summary

    from prepare_dataset import build_dataset, get_dataset_paths, normalize_data
    from custom_dataset import DistributedH5Dataset
    # import custom_callbacks

    device = torch.device(
        # 'cpu'
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Check configuration file validity
    # train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    set_seed(train_config['seed'])

    # Instantiate models
    model = instantiate(model_config,'model')
    print('-- Model :', model)

    # Instantiate criterion
    criterion = instantiate(train_config,'criterion')
    print('-- Criterion :', criterion)

    # Instantiate optimizer
    optimizer = instantiate(train_config,'optimizer', model.parameters(), lr = train_config['lr'])
    print('-- Optimizer :', optimizer)

    # Instantiate scheduler
    scheduler = instantiate(train_config,'scheduler', optimizer)
    print('-- Scheduler :', scheduler)

    # Instantiate evaluation criteria
    eval_criteria = weak_instantiate_all(train_config['eval_criteria'])
    print('-- Evaluation Criteria :')
    if len(eval_criteria):
        for key, val in eval_criteria.items():
            print('  --', key, ':', val)
    else :
        print('  No evaluation criteria.')
        
    # Instantiate callbacks
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

    # print(callbacks_arguments)
    build_dataset()
    train_loader, val_loader = get_dataset_paths(
        ['train','val'],
        split = train_config['splits']
    )
    train_loader    = DistributedH5Dataset(
        train_loader,
        buffer_size             = 32*train_config['batch_size'],
        input_cols              = model_config['input'],
        output_cols             = model_config['output'],
        task                    = train_config['task'],
        remove_mass_pt_window   = model_config['remove_mass_pt_window'],
        return_weights          = train_config['sample_weight'],
        preprocess_data         = lambda data, features = model_config['input']: normalize_data(data, features),
        preprocess_targ         = None,
    )
    val_loader    = DistributedH5Dataset(
        val_loader,
        buffer_size             = 32*train_config['batch_size'],
        input_cols              = model_config['input'],
        output_cols             = model_config['output'],
        task                    = train_config['task'],
        remove_mass_pt_window   = model_config['remove_mass_pt_window'],
        return_weights          = train_config['sample_weight'],
        preprocess_data         = lambda data, features = model_config['input']: normalize_data(data, features),
        preprocess_targ         = None,
    )
    os.makedirs(os.path.join(args.test_dir, 'models'), exist_ok=True)
    data = next(iter(train_loader))[0]
    print(
        '-- Model Summary :',
        get_summary(
            model,
            data,
            dest = os.path.join(args.test_dir, 'models', 'summary')
        )
    )
    train_loader = DataLoader(
        train_loader, 
        batch_size          = train_config['batch_size'],
        num_workers         = os.cpu_count(),
        persistent_workers  = True,
        pin_memory          = device == torch.device('cuda'),
    )
    val_loader = DataLoader(
        val_loader, 
        batch_size          = train_config['batch_size'],
        num_workers         = os.cpu_count(),
        # persistent_workers  = True,
        pin_memory          = device == torch.device('cuda'),
    )
    
    print('-- Using dataset split :', train_config['splits'])
    print('  -- Train      :', len(train_loader.dataset))
    print('  -- Validation :', len(val_loader.dataset))
    print('-- Sample weight:', train_config['sample_weight'])
    history = train(
        model,
        train_dataloader    = train_loader,
        eval_dataloader     = val_loader,
        criterion           = criterion,
        eval_criteria       = eval_criteria,
        optimizer           = optimizer,
        scheduler           = scheduler,
        epochs              = train_config['epochs'],
        patience            = train_config['patience'],
        update_limit        = False,
        sample_weight       = train_config['sample_weight'],
        clip_limit          = 0.5,
        top_dirname         = args.test_dir,
        device              = device,
        evaluate_training   = False,
        saving_steps        = 'log',
        show_pbar           = None if args.no_pbar else 'external',
        callbacks           = callbacks,
        callbacks_arguments = callbacks_arguments,
    )
    print('-- Model :', model)
    