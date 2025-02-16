#!/bin/bash

# Create dataset directories
mkdir -p data/train data/val

# Generate training set
python dataset_generator/dataset_generator.py \
    --output_dir data/train \
    --num_samples 10000 \
    --img_size 224 \
    --font_dir /Library/Fonts

# Generate validation set
python dataset_generator/dataset_generator.py \
    --output_dir data/val \
    --num_samples 1000 \
    --img_size 224 \
    --font_dir /Library/Fonts
