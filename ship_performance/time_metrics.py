import sys, os
import torch
from PIL import Image
import numpy as np
import pandas as pd 
import torch.utils.benchmark as benchmark

????????????????????????????????????

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Local
from kan_utils.fasterkan import FasterKAN
from kan_utils.checkpoint_configs import  load_checkpoint, deconstruct_dir
from kan_utils.experiment_eval import save_attributes
from SkinCancerDataset import *
from quantization.quant_fasterkan import FixedPointFasterKAN, FloatWrapperModule
from quantization.fx_quant import fx_quantize_model

device = torch.device("cpu") 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

top_dir = os.path.dirname(__file__)

dataset_path = r"../Dataset"
csv_path = os.path.join(dataset_path, "HAM10000_metadata.csv") # NOTE: Need to change manually the extension of the file to csv after downloading
csv_test_path = os.path.join(dataset_path, "ISIC2018_Task3_Test_GroundTruth.csv") # NOTE: Need to change manually the extension of the file to csv after downloading
image_dir = os.path.join(dataset_path, "HAM10000_images")
root_dir = os.path.join(dataset_path, "Pretrained")

model_pths= []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if 'epoch_best' in root and os.path.splitext(file)[1] == '.pth':
            model_pths.append(os.path.join(root, file))

x_dim, y_dim, channel_size = 64, 64, 3 
output_classes = 7

image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
df = pd.read_csv(csv_path)
df = df[df['image_id'].isin(image_files)]

dataset = SkinCancerDataset(
    root            = image_dir,
    csv_path        = df, 
    output_classes  = output_classes,
    transform       = get_basic_transform(x_dim, y_dim, channel_size),
)
loader = torch.utils.data.DataLoader(dataset, 1, num_workers=os.cpu_count())

data, _ = next(iter(loader))
data = data.to(device)

model_dict = {
    'Pretrained'                        : FasterKAN,
    'Custom-Quantizer/16_bits'          : FixedPointFasterKAN,
    'Custom-Quantizer/mixed_16_bits'    : FixedPointFasterKAN,
    'Custom-Quantizer/12_bits'          : FixedPointFasterKAN,
    'Custom-Quantizer/mixed_12_bits'    : FixedPointFasterKAN,
    'FX-Quantizer'                      : FasterKAN,
    'Custom-Quantizer/8_bits'           : FixedPointFasterKAN,
    'Custom-Quantizer/mixed_8_bits'     : FixedPointFasterKAN,
    'Custom-Quantizer/4_bits'           : FixedPointFasterKAN,
}
repr_dict = {
    'Pretrained'                        : 'fp32',
    'Custom-Quantizer/16_bits'          : 'int16',
    'Custom-Quantizer/mixed_16_bits'    : 'mixed16',
    'Custom-Quantizer/12_bits'          : 'int12',
    'Custom-Quantizer/mixed_12_bits'    : 'mixed12',
    'FX-Quantizer'                      : 'qint8',
    'Custom-Quantizer/8_bits'           : 'int8',
    'Custom-Quantizer/mixed_8_bits'     : 'mixed8',
    'Custom-Quantizer/4_bits'           : 'int4',
}
def model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt") / 1e6
    os.remove("tmp.pt")
    return f"{size:.2f} MB"

def initialize_model(root_dir, dimension, grid_size, lr, sched, optim, criterion, grid_min, grid_max, 
                    inv_denominator, x_dim, y_dim, channel_size, seed, model_type = FasterKAN):
    """
    Initialize the FasterKAN model with specified parameters.

    Args:
        root_dir (str): The root directory where model checkpoint folders are located.
        dimension (list or tuple): List or tuple specifying the number of units in each hidden layer.
        grid_size (int): Size of the grid.
        lr (float): Learning rate.
        sched (str): Scheduler to use (e.g. 'ReduceOnPlateau', 'ExponentialLR').
        optim (str): Optimizer to use (e.g., 'SGD', 'Adam').
        criterion (str): Loss function to use (e.g., 'CrossEntropyLoss', 'MSELoss').
        grid_min (float): Minimum value for the grid.
        grid_max (float): Maximum value for the grid.
        inv_denominator (float): Inverse denominator for the model.
        x_dim (int): Width of the input data.
        y_dim (int): Height of the input data.
        channel_size (int): Number of channels in the input data.
        seed (int): The seed used for reproductibility.

    Returns:
        model (torch.nn.Module): Initialized FasterKAN model.
        file_path (str): Path to the saved model attributes.
    """
    sched_str = str(sched)
    optim_str = str(optim)
    criterion_str = str(criterion)

    model = model_type(
        hidden_layers=dimension,
        num_grids=grid_size,
        grid_min=grid_min,
        grid_max=grid_max,
        inv_denominator=inv_denominator
    ).cpu()

    # Save model attributes and return the directory path
    file_path = None
    if root_dir is not None:
        file_path = save_attributes(
            model,
            root_dir=root_dir,
            dimension_list=dimension,
            grid_size=grid_size,
            lr=lr,
            sched=sched_str,
            optim=optim_str,
            criterion=criterion_str,
            grid_min=grid_min,
            grid_max=grid_max,
            inv_denominator=inv_denominator,
            x_dim=x_dim,
            y_dim=y_dim,
            channel_size=channel_size,
            seed=seed
        )
    model.to(device)

    return model, file_path

# Compare takes a list of measurements which we'll save in results.
results = []
model_sizes = {}
for model_pth in model_pths:
    seed, criter, optim, sched, lr, dim_list, grid_size, grid_min, grid_max, inv_denominator = deconstruct_dir(os.path.dirname(os.path.dirname(model_pth)))
    
    for model_type, model_class in model_dict.items():
        model_type_path = model_pth.replace('Pretrained', model_type)
        model, _ = initialize_model(
            root_dir=None,
            dimension=dim_list,
            grid_size=grid_size,
            lr=lr,
            sched=sched,
            optim=optim,
            criterion=criter,
            grid_min=grid_min,
            grid_max=grid_max,
            inv_denominator=inv_denominator,
            x_dim=x_dim,
            y_dim=y_dim,
            channel_size=channel_size,
            seed=seed,
            model_type=model_class
        )
        if model_type == 'FX-Quantizer':
            model = fx_quantize_model(model, loader, device)
            
        try :
            model, *_ = load_checkpoint(model, checkpoint_path=model_type_path, device=device) 
        except:
            continue
        
        if isinstance(model, FixedPointFasterKAN):
            model = FloatWrapperModule(model)
            
        model_sizes.update({model_type_path : model_size(model)})
        
        # label and sub_label are the rows
        # description is the column
        for num_threads in range(1,os.cpu_count()+1):
            results.append(benchmark.Timer(
                stmt='model(data)',
                globals={'model': model, 'data' : data},
                num_threads=num_threads,
                label='Inference',
                sub_label=f'Scale={inv_denominator}',
                description=repr_dict[model_type],
            ).blocked_autorange(min_run_time=10))

compare = benchmark.Compare(results)
tmp = sys.stdout

# Evaluate model sizes
f_rslt = os.path.join(top_dir,'mem_results.txt')
with open(f_rslt, 'w') as sys.stdout:
    print("\nModel Sizes")
    print("=" * 40)
    for i, (pth, size) in enumerate(model_sizes.items()):
        print(f"Model {i} : path = '{pth}'")
        print(f"            size = {size}")
    print("=" * 40)
sys.stdout = tmp

f_rslt = os.path.join(top_dir,'time_results.txt')
with open(f_rslt, 'w') as sys.stdout:
    compare.print()
    
sys.stdout = tmp
print(f'Results saved in {f_rslt}')