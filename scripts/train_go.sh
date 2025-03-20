#!/bin/bash

# Example command to run the training script with specific parameters
# This script will train the unsupervised model, which is used as a starting point to fine-tune downstream tasks.
python src/run.py \
    --epochs 1 \
    --lr 3e-05 \
    --pool "mean" \
    --max_length 512 \
    --batch_size 100 \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --data_path "data/gene_go_triplet.csv" \
    --task_type "unsupervised" \
    --save_model_path "checkpoints/LitGene_GO_" \
    --test_split_size 0.15 \
    

