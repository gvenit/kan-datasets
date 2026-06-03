#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Training script for Skin Cancer Dataset (Simplified).')
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

    import torch
    from torch.utils.data import DataLoader

    from kan_utils.config import *
    from kan_utils.dataset import smart_split_dataset
    from kan_utils.training import train
    from kan_utils.utils import set_seed
    from kan_utils.performance import get_summary
    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset, get_groups
    from custom_dataset import SkinCancerDataset, get_extra_transforms
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    set_seed(train_config['seed'])

    model = instantiate(model_config,'model')
    print('-- Model :', model)

    criterion = instantiate(train_config,'criterion')
    print('-- Criterion :', criterion)

    optimizer = instantiate(train_config,'optimizer', model.parameters(), lr = train_config['lr'])
    print('-- Optimizer :', optimizer)
    
    scheduler = instantiate(train_config,'scheduler', optimizer)
    print('-- Scheduler :', scheduler)

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

    # Load dataset and create train/val split
    df = expand_df_labels(normalize_dataset(build_dataset()))
    full_dataset = SkinCancerDataset(
        df,
        input_cols=['Image'],
        output_cols={
            'classifier': [col for col in df.columns if col.startswith('Lesion_')]
        },
        input_img_dims=model_config.get('input_img_dim', (3, 64, 64)),
        path_cols={'image': ['Image']},
        seed=train_config['seed']
    )

    groups = get_groups(
        exclude     = train_config['exclude_groups'] if 'exclude_groups' in train_config.keys() else None
    )
    
    # Split into train and validation
    train_loader, val_loader, *_ = smart_split_dataset(
        splits          = train_config['splits'],
        full_dataset    = full_dataset,
        groups          = groups,
        seed            = train_config['seed']
    )
    training_subdir = args.test_dir
    os.makedirs(training_subdir, exist_ok=True)

    train_loader = DataLoader(
        train_loader,
        batch_size          = train_config['batch_size'],
        num_workers         = os.cpu_count() if os.cpu_count() and os.cpu_count() > 1 else 0,
        persistent_workers  = True if os.cpu_count() and os.cpu_count() > 1 else False,
        pin_memory          = device == torch.device('cuda'),
    )
    val_loader = DataLoader(
        val_loader,
        batch_size          = train_config['batch_size'],
        num_workers         = os.cpu_count() if os.cpu_count() and os.cpu_count() > 1 else 0,
        persistent_workers  = False,
        pin_memory          = device == torch.device('cuda'),
    )

    sample_data, sample_target = next(iter(train_loader))

    print(
        '-- Model Summary :',
        get_summary(
            model,
            sample_data[0:1],
            dest = os.path.join(training_subdir, 'summary')
        )
    )

    print('-- Using dataset split :', train_config['splits'])
    print('  -- Train      :', len(train_loader.dataset))
    print('  -- Validation :', len(val_loader.dataset))

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
        sample_weight       = train_config.get('sample_weight', 2.0),
        top_dirname         = training_subdir,
        device              = device,
        evaluate_training   = True,
        saving_steps        = 'log',
        show_pbar           = 'external',
        update_limit        = 10,
        callbacks           = callbacks,
        callbacks_arguments = callbacks_arguments,
    )
    print('-- Model :', model)
