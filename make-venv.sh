#!/bin/bash

python3 -m venv venv

source venv/bin/activate

pip install -U numpy pandas pillow torch torchinfo torchvision \
        tqdm matplotlib seaborn albumentationsx kaggle kagglehub \
        scikit-learn torchmetrics