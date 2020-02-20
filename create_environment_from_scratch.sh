#!/bin/bash

conda create -n gale python=3
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install colorlog numpy pandas tqdm matplotlib seaborn
pip install sigopt tensorboardx darwin-py