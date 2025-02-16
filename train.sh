#!/bin/bash

# Create checkpoints directory
mkdir -p checkpoints

python train.py \
    --train_clean_dir data/train/clean \
    --train_corrupted_dir data/train/corrupted \
    --val_clean_dir data/val/clean \
    --val_corrupted_dir data/val/corrupted \
    --max_epochs 100 \
    --accelerator cpu \
    --devices 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --num_workers 4 \
    --img_size 224 \
    --patch_size 16 \
    --decoder_layers 6 \
    --decoder_heads 8
