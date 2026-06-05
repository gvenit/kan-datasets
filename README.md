# kan-datasets

This repository implements our research framework for Radial Basis Function Kolmogorov-Arnold Networks (RBF-KANs), developed as part of our work on edge-oriented computing for IoT devices. As established in recent literature, Kolmogorov-Arnold Networks (KANs) have set a new standard in machine learning tasks by prevailing over traditionally deployed multilayer perceptrons (MLPs), offering enhanced interpretability through learnable activation functions on network edges rather than fixed activation functions on nodes. However, standard KANs come with increased computational complexity and memory footprint, making deployment on resource-constrained edge devices challenging.

RBF-KANs address this limitation by replacing conventional B-spline basis functions with Radial Basis Functions (RBFs), achieving considerable model size reduction while maintaining the key advantages of KANs. The RBF parameterization enables more efficient execution through simpler kernel evaluations. This makes RBF-KANs particularly suitable for edge computing scenarios where computational resources, memory, and power are limited.

Our framework serves as the software counterpart to a fully pipelined, runtime-configurable hardware IP core we designed for executing RBF-KANs on all-programmable systems-on-chip (APSoC). The repository provides:

- Software implementations of RBF-KAN models for training and validation
- Quantization tools to prepare models for hardware deployment
- Evaluation pipelines across multiple datasets
- Configurable architectures (layer sizes, grid resolutions, RBF types)

The models trained with this framework can be deployed on our hardware accelerator, which achieves up to 43.6× speedup compared to commercial edge CPUs with significantly lower power consumption, enabling real-time inference and latency-sensitive neural network deployment on IoT devices.

## Running the Framework

In each dataset folder you can find a main script to run the whole pipeline from downloading each dataset, training the KAN model and testing it. Before running the script make sure you create a virtual environment, using the `make-venv.sh` script in the root directory of the repository. This will create a virtual environment and install the required dependencies. You can activate the virtual environment using the following command:

```bash
cd /path/to/repository
chmod +x make-venv.sh
./make-venv.sh
source venv/bin/activate
```

After activating the virtual environment, you can run the main script in each dataset folder to start the pipeline. Make sure to adjust the configurable parameters in the script as needed before running it.

```bash
cd /path/to/dataset_folder
chmod +x run-kan.sh
./run-kan.sh
```

### Configurable Parameters

The main script exposes several configurable parameters that control the experiment setup, model architecture, and optimization process.

#### Experiment Settings

- `TEST_VERSION`: Experiment identifier used for logging, checkpoint naming, and result tracking. 
- `SEED`: Random seed for reproducibility.

#### Data Processing

Note: These parameters may not be available to all experiments.
- `WITH_LOGITS`: Enables training with logits instead of probabilities where applicable.
- `RESIZE`: Optional image resize resolution specified as `"H W"` (e.g., `"16 16"`). Leave empty to use the default input size.

#### Model Architecture

- `LAYERS`: Hidden layer sizes. For example, `"256 256"` creates two hidden layers with 256 neurons each. 
- `NUM_GRIDS`: Number of grid points used by the KAN layers. Multiple values can be specified for different layers (e.g., `"64 128"` for two KAN layers). If only one value is provided, it will be applied to all KAN layers.
- `GRID_MIN`: Lower bound of the grid range. Multiple values can be specified for different layers (e.g., `"-1.0 -0.5"` for two KAN layers). If only one value is provided, it will be applied to all KAN layers.
- `GRID_MAX`: Upper bound of the grid range. Multiple values can be specified for different layers (e.g., `"1.0 0.5"` for two KAN layers). If only one value is provided, it will be applied to all KAN layers.
- `SCALE`: Scaling factor applied to the grid representation. Multiple values can be specified for different layers (e.g., `"1.0 0.5"` for two KAN layers). If only one value is provided, it will be applied to all KAN layers.

Note: The following two parameters are only applicable to certain datasets
- `DYNAMIC`: Generates grid ranges and scale dynamically based on the input data distribution when enabled using a small subnetwork.
- `USE_V2`: Uses the second implementation/version of the KAN layer when enabled that has a different learnable grid per input dimension.

#### Training Hyperparameters

- `EPOCHS`: Number of training epochs.
- `BATCH`: Batch size for training.
- `LR`: Learning rate for the optimizer.

#### Optimization Settings

- `OPTIMIZER`: Choice of optimizer (e.g., `Adam`, `RMSprop`).
- `WEIGHT_DECAY`: L2 regularization strength.
- `MOMENTUM`: Momentum factor for optimizers that support it (e.g., `SGD`).

Note: For the experiments that support learning rate scheduling, the following parameters are also available:
- `LR_FACTOR`: Multiplicative factor used when reducing the learning rate.
- `LR_PATIENCE`: Number of epochs without improvement before the learning rate scheduler reduces the learning rate.

#### Regularization and Normalization Parameters
Note: The following parameters are only applicable to certain experiments:
- `DROPOUT`: Dropout probability applied to KAN layers.
- `LINEAR_DROPOUT`: Dropout probability applied to linear layers.
- `NO_NORMALIZE`: Disables feature normalization when set to `1`. This should typically remain `0` during training.
- `NO_NORMALIZE_RBF`: Disables RBF normalization when set to `1`. By default, RBF normalization is enabled.

#### Additional Options

- `MODE`: Use different basis functions for the KAN layers (e.g. `RSWAFF`, `tanh`).
- `RESIDUAL`: Whether to use residual connections in the KAN layers.

### How to Modify Script Configurations

To modify the configurations for the KAN experiments, you can edit the `run-kan.sh` script located in each dataset folder. This script contains the parameters mentioned above, which you can adjust according to your requirements. But if you want to have more control over the pipeline you can adjust the individual `create_configs.py` files. If you need to modify the data preprocessing steps or apply different data cleaning techniques, you can edit the `prepare_data.py` script. 

Note: Some experiments may have extra files for their pipeline as each experiment is different. Also, some experiments require their custom dataset objects to be implemented. 

#### Sweep Scripts

Some experiments have sweep scripts `run-kan-sweep.sh` that allow you to run multiple configurations in a single execution. These scripts typically loop over all different values of parameters in the `run-kan.sh` script and execute the pipeline for each combination. You can modify these sweep scripts to include the specific parameter ranges you want to explore in your experiments.

## The `kan_utils` Module

The `kan_utils` module provides a comprehensive set of utilities for working with Radial Basis Function based Kolmogorov Arnold Networks (RBF-KANs), including configuration management, metrics, model architectures (both standard and quantized), performance tracking, quantization tools, training utilities, and helper functions.

### Module Structure

```
kan_utils/
├── config/              # Configuration management
├── metrics/             # Loss functions and evaluation metrics
├── models/              # KAN model architectures
│   └── quantized/       # Quantized model variants
├── performance/         # Performance tracking and summarization
├── quantization/        # Quantization utilities
├── training/            # Training and evaluation loops
├── callbacks.py         # Training callbacks
├── dataset.py           # Dataset utilities
└── plotter.py           # Visualization tools
```