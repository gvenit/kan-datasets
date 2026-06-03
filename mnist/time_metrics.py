#!/usr/bin/env python3
import os
import torch
import platform

def model_size(mdl, path = None):
    if path is None:
        torch.save(mdl.state_dict(), "tmp.pt")
        size = os.path.getsize("tmp.pt")
        os.remove("tmp.pt")
    else :
        size = os.path.getsize(path)
    return size

def optimize_torch_for_hardware():
    # 1. Detect Architecture
    arch = platform.machine().lower()
    is_arm = 'arm' in arch or 'aarch64' in arch

    if is_arm:
        # --- Raspberry Pi / ARM Specifics ---
        # Check for available quantized engines (XNNPACK is generally faster than QNNPACK)
        supported = torch.backends.quantized.supported_engines
        if 'xnnpack' in supported:
            torch.backends.quantized.engine = 'xnnpack'
        elif 'qnnpack' in supported:
            torch.backends.quantized.engine = 'qnnpack'
        
        # Limit threads to match Pi 4 physical cores (prevents over-scheduling)
        torch.set_num_threads(4)
        
        # Set environment variable to reduce "spinning" on ARM (saves battery/heat)
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        
        print(f"[*] ARM detected ({arch}). Applied {torch.backends.quantized.engine} optimization.")
    else:
        # --- x86 / Windows / Intel Specifics ---
        # On modern x86, 'x86' (fbgemm) is the standard high-speed backend
        if 'x86' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'x86'
        
        print(f"[*] Standard architecture detected ({arch}). Using default backends.")

# Execute immediately upon import/run
optimize_torch_for_hardware()

def model_size(mdl, path = None):
    if path is None:
        torch.save(mdl.state_dict(), "tmp.pt")
        size = os.path.getsize("tmp.pt")
        os.remove("tmp.pt")
    else :
        size = os.path.getsize(path)
    return size

if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR  = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Quantize trained FasterKAN models for the CERN dataset.')
    parser.add_argument('-d', '--train-dir', dest='train_dir', default=os.path.join(THIS_DIR, 'train'), help='Top-level training output directory (default: ./train)')
    parser.add_argument('-l', '--limit', dest='limit', type=int, default=-1, help='Limit the number of characters of the hashes shown in the figures.')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')
    parser.add_argument('--hardtanh', dest='hardtanh', action='store_true', default=False, help='Use integer Hardtanh approximation (RSWAFF mode only)')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all', dest='all_models', action='store_true', default=False, help='Process all models in the train directory (default behaviour)')
    sel.add_argument('-m','--model', dest='model_name', type=str, default=None, help='Process a specific run by folder name, e.g. test_0 or <hash>/test_0')

    args = parser.parse_args()

    # ------------------------------------------------------------------ 
    from torch.utils.data import DataLoader
    import torch.utils.benchmark as benchmark
    import pandas as pd 
    import albumentations as A

    from kan_utils.config.config import load_config, instantiate
    from kan_utils.utils import load_model
    from kan_utils.quantization import FixedPointFasterKAN, FloatWrapperModule
    
    from prepare_dataset import build_dataset, get_dataset
    from custom_dataset import MNISTDataset

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) 
    # device = "cpu"
    # device = torch.device("cpu") 

    model_pths= []
    top_search_path = args.train_dir
    if args.model_name is not None:
        top_search_path = os.path.join(
            top_search_path, 
            args.model_name,
        )
        if not os.path.exists(top_search_path):
            raise ValueError(f'Provided model does not exist inside training directory "{args.train_dir}"')
        
    for root, dirs, files in os.walk(top_search_path):
        for file in files:
            if args.epoch in file and os.path.splitext(file)[1]  in ('.pth','.pt'):
                model_pths.append(os.path.join(root, file))

    build_dataset() 
    test_data, test_labels = get_dataset('test')
    
    preprocess_data = lambda train_config : A.Compose([
            *([] if 'resize' not in train_config.keys() else [A.Resize(*train_config['resize'])]),
            A.Normalize(normalization = 'min_max_per_channel'), 
            # A.Normalize(normalization = 'standard'), #ImageNet mean/std
            A.ToTensorV2(),    
        ], 
        telemetry   = False,
        seed        = train_config['seed'],
    )
    
    # Compare takes a list of measurements which we'll save in results.
    results = []
    model_sizes = []
    with torch.no_grad():
        for model_pth in model_pths:
            # Locate configuration files
            model_top_dir = os.path.dirname(model_pth)
            while 'config' not in os.listdir(model_top_dir):
                model_top_dir = os.path.dirname(model_top_dir)
            
            model_hash    = os.path.basename(os.path.dirname(model_top_dir))
            model_version = os.path.basename(model_top_dir)
            
            model_config = load_config(os.path.join(model_top_dir, 'config', 'model.json'))
            train_config = load_config(os.path.join(model_top_dir, 'config', 'train.json'))
            
            base_model   = instantiate(model_config, 'model')
            
            if (os.sep + 'models') in model_pth:
                model = base_model
                model_dr = 'fp32'
                
            elif (os.sep + 'quantized_models') in model_pth:
                if isinstance(base_model, torch.nn.Sequential) and len(base_model) == 1:
                    base_model = next(base_model.children())
                model = FixedPointFasterKAN(base_model, hardtanh=args.hardtanh)
                model_dr = os.path.basename(os.path.dirname(model_pth))
                
            test_loader = MNISTDataset(
                test_data, test_labels,
                task            = train_config['task'],
                return_weights  = train_config['sample_weight'],
                return_key      = True,
                preprocess_data = lambda **kwargs: preprocess_data(train_config)(**kwargs)[list(kwargs.keys())[0]],
                flatten         = model_config['flatten'],
            )
            test_loader = DataLoader(
                test_loader,
                batch_size  = train_config['batch_size'],
                num_workers = os.cpu_count(),
                pin_memory  = False,
            )
            
            # print('-- Model :', model)
            print('-- Benchmarking', model_hash[:args.limit], model_version, model_dr)
                
            load_model(model, model_pth, "cpu")
            
            # if isinstance(model, FixedPointFasterKAN):
            #     model = FloatWrapperModule(model)
                
            # label and sub_label are the rows
            # description is the column
            model_results = []
            for compiled in [False, True]:
                if compiled:
                    try :
                        model.compile()
                    except Exception as e:
                        print(e)
                        continue
                    
                for device in devices:
                    model.to(device)
                    for num_threads in range(1,torch.get_num_threads()+1):
                        print(32*'--')
                        print('  -- Threads / Device :', num_threads, '/', device, f'{"(compiled)" if compiled else ""}')
                        timer = benchmark.Timer(
                            stmt        = 'model(data)',
                            globals     = {'model': model, 'data' : next(iter(test_loader))[0].to(device)},
                            num_threads = num_threads,
                            label       = f'{model_hash[:args.limit]}/{model_version}',
                            sub_label   = device,
                            description = f'Latency ({model_dr})',
                            env         = "compiled" if compiled else "eager mode",
                        )
                        if device == 'cpu':
                            model_results.append(
                                timer.adaptive_autorange(min_run_time=0.5, max_run_time=10)
                            )
                        else :
                            model_results.append(
                                timer.blocked_autorange(min_run_time=1)
                            )
                        print(model_results[-1])
                        print('    -- Mean :', model_results[-1].mean)
                        if device != 'cpu':
                            break
                        
                    # model_sizes.append({
                    #     'Hash'      : model_hash,
                    #     'Version'   : model_version,
                    #     'Data_repr' : model_dr,
                    #     'Size'      : model_size(model, model_pth),
                    #     # 'Device'    : device,
                    #     # 'Compiled'  : compiled,
                    #     # 'Latency'   : str(model_results[-1]),
                    # })
            results.extend(model_results)
    
    results = list(filter(lambda x: x is not None, results))
    compare = benchmark.Compare(results)

    # # Evaluate model sizes
    # f_rslt = os.path.join(args.train_dir,'comparison',f'mem_results_{pd.Timestamp.now().strftime("%Y-%m-%d-%X")}.txt')
    # model_sizes = pd.DataFrame.from_records(model_sizes)
    # # model_sizes = pd.DataFrame.from_records(model_sizes, exclude=['Latency'])
    # model_sizes.to_csv(f_rslt, index=False)
    # print(f'Memory sizes saved in {f_rslt}')
    
    # print(results)

    tmp = sys.stdout
    f_rslt = os.path.join(args.train_dir,'comparison',f'time_results_{pd.Timestamp.now().strftime("%Y-%m-%d-%X")}.log')
    
    # try :
    compare.print()
    # except :
    #     pass
        # print(results)
        
    with open(f_rslt, 'w') as sys.stdout:
        try :
            compare.print()
        except :
            print(results)
        
    sys.stdout = tmp
    print(f'Results saved in {f_rslt}')