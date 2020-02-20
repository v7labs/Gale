#!/bin/bash

conda create -y -n gale python=3.7.4 colorlog numpy pandas tqdm matplotlib seaborn pytorch torchvision cudatoolkit=10.1 -c pytorch 
source activate gale
pip install sigopt tensorboardx darwin-py wandb psutil
