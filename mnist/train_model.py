#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Training script for MNIST Dataset.')
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
    import albumentations as A

    from kan_utils.config import *
    from kan_utils.dataset import smart_split_dataset
    from kan_utils.training import train
    from kan_utils.utils import set_seed, apply_to_tensor
    from kan_utils.performance import get_summary

    from prepare_dataset import build_dataset, get_dataset, get_groups, normalize_data
    from custom_dataset import *
    import custom_model
    # import custom_callbacks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals(custom_model))
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

    data, labels = get_dataset('train_val')

    preprocess_data = A.Compose([
            A.HorizontalFlip(p=train_config['probability']),
            A.VerticalFlip(p=train_config['probability']),
            A.ShiftScaleRotate(scale_limit=(-0.2,0),rotate_limit=(-90,90), p=train_config['probability']),
            A.SafeRotate(limit=(-90,90), p=train_config['probability']),
            *([] if 'resize' not in train_config.keys() else [A.Resize(*train_config['resize'])]),
            A.Normalize(normalization = 'min_max_per_channel'), 
            # A.Normalize(normalization = 'standard'), #ImageNet mean/std
            A.ToTensorV2(),    
        ], 
        telemetry   = False,
        seed        = train_config['seed'],
    )
    # resize = train_config.get('resize', None)
    # if resize is not None:
    #     import torch.nn.functional as F
    #     def preprocess_data(images):
    #         t = torch.from_numpy(np.array(images, dtype=np.float32)).unsqueeze(1)
    #         t = F.interpolate(t, size=tuple(resize), mode='bilinear', align_corners=False)
    #         return normalize_data(t.squeeze(1).numpy())
    # else:
    #     def preprocess_data(images):
    #         return normalize_data(images)

    # Create iterable train and val datasets
    train_loader, val_loader = smart_split_dataset(
        splits          = train_config['splits'],
        full_dataset    = None,
        groups          = get_groups(),
        seed            = train_config['seed']
    )
    train_loader = MNISTDataset(
        data[train_loader], labels[train_loader],
        task                    = train_config['task'],
        return_weights          = train_config['sample_weight'],
        preprocess_data         = lambda **kwargs: preprocess_data(**kwargs)[list(kwargs.keys())[0]],
        preprocess_targ         = None,
        flatten                 = model_config['flatten'],
    )
    val_loader = MNISTDataset(
        data[val_loader], labels[val_loader],
        task                    = train_config['task'],
        return_weights          = train_config['sample_weight'],
        preprocess_data         = lambda **kwargs: preprocess_data(**kwargs)[list(kwargs.keys())[0]],
        preprocess_targ         = None,
        flatten                 = model_config['flatten'],
    )
    
    os.makedirs(os.path.join(args.test_dir, 'models'), exist_ok=True)
    data = next(iter(train_loader))[0]
    print(
        '-- Model Summary :',
        get_summary(
            model,
            data,
            dest = os.path.join(args.test_dir, 'models', 'summary'),
            depth = 5,
        )
    )

    train_loader = DataLoader(
        train_loader, 
        shuffle             = True,
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
        update_limit        = 100,
        sample_weight       = train_config['sample_weight'],
        # clip_limit          = 0.5,
        top_dirname         = args.test_dir,
        device              = device,
        evaluate_training   = False,
        saving_steps        = 'log',
        show_pbar           = None if args.no_pbar else 'external',
        callbacks           = callbacks,
        callbacks_arguments = callbacks_arguments,
    )
    print('-- Model :', model)