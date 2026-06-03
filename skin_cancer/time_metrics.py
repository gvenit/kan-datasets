#!/usr/bin/env python3
import os
import torch
import platform

def model_size(mdl, path = None):
    try:
        if path is None:
            torch.save(mdl.state_dict(), "tmp.pt")
            size = os.path.getsize("tmp.pt")
            os.remove("tmp.pt")
        else :
            size = os.path.getsize(path)
    except:
        size = -1
    finally:    
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


def model_size(mdl, path = None):
    try:
        if path is None:
            torch.save(mdl.state_dict(), "tmp.pt")
            size = os.path.getsize("tmp.pt")
            os.remove("tmp.pt")
        else :
            size = os.path.getsize(path)
    except:
        size = -1
    finally:    
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
    parser.add_argument('--epoch', dest='epoch', action='append')
    parser.add_argument('--stage', dest='stage', type=int, default=1)
    parser.add_argument('--fuse', dest='fuse', action='store_true')
    parser.add_argument('--repr', dest='repr', action='append', help='Process a specific head of a model by folder name, e.g. test_0 or <hash>')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all', dest='all_models', action='store_true', default=False, help='Process all models in the train directory (default behaviour)')

    group = sel.add_argument_group()
    group.add_argument('-m','--model', dest='model_hash', action='append', help='Process a specific model by hash')
    group.add_argument('-t','--train', dest='train_hash', action='append', help='Process a specific training by hash')
    group.add_argument('-v','--version', '--test-version', dest='test_version', action='append', help='Process a specific training by hash')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all-heads', dest='all_heads', action='store_true', default=False, help='Process all heads of a model (default behaviour)')
    sel.add_argument('--head', dest='head', action='append', help='Process a specific head of a model by folder name, e.g. test_0 or <hash>')

    args = parser.parse_args()

    # ------------------------------------------------------------------ 
    from warnings import warn
    import pandas as pd 
    from torch.utils.data import DataLoader
    import torch.utils.benchmark as benchmark

    from kan_utils.config import *
    from kan_utils.models import fuse_faster_kan
    from kan_utils.utils import load_model, to
    from kan_utils.quantization import FixedPointFasterKAN
    
    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
    from build_model import get_train_config_path, get_model_config_path, build_model, get_training_subdir
    from custom_dataset import SkinCancerDataset, get_extra_transforms
    from prepare_dataset import build_dataset, get_dataset
    import custom_model, extract_statistics

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) 
    # device = "cpu"
    # device = torch.device("cpu") 
    
    # Execute immediately upon import/run
    optimize_torch_for_hardware()

    # Find model hashes
    model_versions = [
        (model_hash, os.path.splitext(test_version)[0]) 
            for model_hash in os.listdir(
                os.path.join(args.train_dir, 'model_config')
            )
            for test_version in os.listdir(
                os.path.join(args.train_dir, 'model_config', model_hash)
            )
    ]
    train_versions = [
        (train_hash, os.path.splitext(test_version)[0]) 
            for train_hash in os.listdir(
                os.path.join(args.train_dir, 'train_config')
            )
            for test_version in os.listdir(
                os.path.join(args.train_dir, 'train_config', train_hash)
            )
    ]
    runs = []

    for train_version in train_versions:
        for model_version in model_versions:
            if train_version[-1] != model_version[-1]:
                continue
            curr_run = (
                model_version[0],
                *train_version,
            )
            pth = get_training_subdir(
                training_stage  = args.stage,
                model_hash      = curr_run[0],
                train_hash      = curr_run[1],
                test_version    = curr_run[2],
                top_dir         = args.train_dir
            )
            pth = os.path.join(
                get_training_subdir(
                    training_stage  = args.stage,
                    model_hash      = curr_run[0],
                    train_hash      = curr_run[1],
                    test_version    = curr_run[2],
                    top_dir         = args.train_dir
                ), 'models'
            )
            if os.path.exists(pth):
                for epoch in os.listdir(pth): 
                    if args.epoch and sum(_ in epoch for _ in args.epoch) == 0:
                        continue
                    
                    curr_run += os.path.splitext(epoch)[0],
                    
                    if args.all_models:
                        runs.append(curr_run)
                    else :
                        if args.model_hash and curr_run[0] not in args.model_hash:
                            continue
                        
                        if args.train_hash and curr_run[1] not in args.train_hash:
                            continue
                        
                        if args.test_version and curr_run[2] not in args.test_version:
                            continue
                        
                        runs.append(curr_run)
    
    if not runs:
        print('No trained models found. Run train_model.py first.')
        sys.exit(0)

    # Compare takes a list of measurements which we'll save in results.
    results = []
    model_sizes = []
    with torch.no_grad():
        for run in runs:
            # Locate configuration files
            training_subdir = get_training_subdir(
                training_stage  = args.stage,
                model_hash      = run[0],
                train_hash      = run[1],
                test_version    = run[2],
                top_dir         = args.train_dir,
            )
            quantized_subdir = os.path.join(training_subdir,'quantized_models')
            train_config = get_train_config_path(
                training_stage  = args.stage,
                train_hash      = run[1],
                test_version    = run[2],
                top_dir         = args.train_dir,
            )
            model_config = get_model_config_path(
                training_stage  = args.stage,
                model_hash      = run[0],
                test_version    = run[2],
                top_dir         = args.train_dir,
            )
            train_config = load_config(train_config, locals=get_locals(extract_statistics))
            model_config = load_config(model_config, locals=get_locals(custom_model))
            
            # base_model   = instantiate(model_config, 'model')
            img_enc = instantiate(model_config,'Image Encoder')
            heads   = {
                head : instantiate(model_config, head)
                    for head in model_config['heads']
            }
            base_model   = build_model(
                img_enc = img_enc,
                **heads,
            )
            base_model.eval()
            # try :
            #     load_model(model, run_path)
            #     model.eval()
            # except Exception as e:
            #     print('  -- Failed')
            #     if args.warn:
            #         warn(str(e))
            #     continue
            
            data = SkinCancerDataset(
                normalize_dataset(expand_df_labels(build_dataset())).head(512),
                input_cols      = model_config['input'],
                output_cols     = model_config['output'],
                input_img_dims  = model_config['input_img_dim'],
                path_cols       = model_config['path_cols'],
                extra_transforms= get_extra_transforms(0.2, *model_config['input_img_dim'][-2:]),
                flatten         = model_config['flatten'] if 'flatten' in model_config.keys() else False,
                return_key      = False,
                return_type     ='dict',
                seed            = train_config['seed']
            )
            data, *_  = data[0]
            data = data.unsqueeze(0)
            
            # Find representation types
            model_drs = ['fp32']
            if os.path.exists(quantized_subdir):
                model_drs.extend(os.listdir(quantized_subdir))
            
            model_results = []
            # for compiled in [True]:
            for compiled in [False, True]:
                for model_dr in model_drs:
                    if args.repr and model_dr not in args.repr:
                        continue
                    for head_name, head in heads.items():
                        if args.head and head_name not in args.head:
                            print('-- Skipping:', head_name)
                            continue
                    
                        if model_dr == 'fp32':
                            run_path = os.path.join(
                                training_subdir,
                                'models',
                                run[-1]
                            )
                            try :
                                load_model(base_model, run_path)
                            except:
                                print(' -- Skipping:', head_name, 'fp32')
                                continue
                                
                            if args.fuse:
                                try :
                                    model = fuse_faster_kan(
                                        img_enc,
                                        head
                                    )
                                except:
                                    print('  -- Fusion failed.')
                                    model = build_model(
                                        img_enc = img_enc,
                                        **{head_name : head}
                                    )
                            else :
                                model = base_model
                        else :
                            run_path = os.path.join(
                                training_subdir,
                                'quantized_models',
                                model_dr,
                                head_name,
                                run[-1],
                                run[-1]
                            )
                            model = fuse_faster_kan(
                                img_enc,
                                head
                            )
                            model = FixedPointFasterKAN(model)
                            try:
                                load_model(model, run_path)
                            except:
                                print(' -- Skipping:', head_name, 'fp32')
                                continue
                            
                        print('-- Benchmarking', run[0][:args.limit], '/', run[1][:args.limit], '/', run[2], '/', run[3], '/', head_name, '/', model_dr, *(['/','compiled'] if compiled else []))
                
                        for device in devices:
                            model.eval().to(device)
                            data = to(data, device)
                            
                            if compiled:
                                try :
                                    model.compile()
                                except Exception as e:
                                    print('  -- Model compilation failed')
                                    print(e)
                                    continue
                                
                            # try:
                            for _ in range(100):
                                _ = model(data)
                            # except Exception as e:
                            #     print('  -- Model execution failed')
                            #     print(e)
                            #     continue
                            
                            for num_threads in range(1,torch.get_num_threads()+1):
                                print(32*'--')
                                print('  -- Threads / Device :', num_threads, '/', device, f'{"(compiled)" if compiled else ""}')
                                timer = benchmark.Timer(
                                    stmt        = 'model(data)',
                                    globals     = {'model': model, 'data' : data},
                                    num_threads = num_threads,
                                    label       = f'{run[0][:args.limit]}/{run[1][:args.limit]}/{run[2]}',
                                    sub_label   = device,
                                    description = f'{head_name} ({model_dr})',
                                    env         = "compiled" if compiled else "eager mode",
                                )
                                if device == 'cpu':
                                    model_results.append(
                                        timer.adaptive_autorange(min_run_time=5, max_run_time=50)
                                    )
                                else :
                                    model_results.append(
                                        timer.blocked_autorange(min_run_time=10)
                                    )
                                print(model_results[-1])
                                print('    -- Mean :', model_results[-1].mean)
                                if device != 'cpu':
                                    break
                        
                        model_sizes.append({
                            'Model'     : run[0],
                            'Training'  : run[0],
                            'Version'   : model_version,
                            'Head'      : head_name,
                            'Data_repr' : model_dr,
                            'Size'      : model_size(model, run_path),
                            # 'Device'    : device,
                            'Compiled'  : compiled,
                            # 'Latency'   : str(model_results[-1]),
                        })
            results.extend(model_results)
    
    results = list(filter(lambda x: x is not None, results))
    compare = benchmark.Compare(results)

    # Evaluate model sizes
    f_rslt = os.path.join(args.train_dir,'comparison',f'mem_results.txt')
    os.makedirs(os.path.dirname(f_rslt), exist_ok=True)
    model_sizes = pd.DataFrame.from_records(model_sizes)
    # model_sizes = pd.DataFrame.from_records(model_sizes, exclude=['Latency'])
    model_sizes.to_csv(f_rslt, index=False)
    print(f'Memory sizes saved in {f_rslt}')
    
    # print(results)

    tmp = sys.stdout
    f_rslt = os.path.join(args.train_dir,'comparison',f'time_table_{pd.Timestamp.now().strftime("%Y-%m-%d-%X")}.log')
    with open(f_rslt, 'w') as sys.stdout:
        try :
            compare.print()
        except :
            # print(results)
            pass
            
    f_rslt = os.path.join(args.train_dir,'comparison',f'time_results_{pd.Timestamp.now().strftime("%Y-%m-%d-%X")}.log')
    with open(f_rslt, 'w') as sys.stdout:
        print(results)
        
    sys.stdout = tmp
    print(f'Results saved in {f_rslt}')
    
    # try :
    compare.colorize()
    compare.print()
    # except :
    #     pass
        # print(results)
        