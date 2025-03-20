#!/bin/bash

# Example command to run the training script with specific parameters
python src/run.py \
    --epochs 3 \
    --lr 3e-05 \
    --pool "mean" \
    --max_length 512 \
    --batch_size 100 \
    --model_name "tumorailab/LitGene_ContrastiveLearning" \
    --data_path "data/combined_solubility.csv" \
    --task_type "classification" \
    --save_model_path "checkpoints/combined_solubility_" \
    --test_split_size 0.15 \
    --val_split_size 0.15
