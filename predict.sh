#!/bin/bash

# Create output directory
mkdir -p cleaned_documents

# Run prediction
python predict.py \
    --model_path checkpoints/document-cleaning-last.ckpt \
    --input_path test_images \
    --output_path cleaned_documents \
    --device cuda
