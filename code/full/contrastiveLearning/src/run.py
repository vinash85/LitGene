import torch
import gc
import pandas as pd
from sklearn.metrics import accuracy_score
from litgene import FineTunedBERT
from utils import process_data
from train import trainer
import argparse

def parse_parameters(
    epochs,
    lr,
    pool,
    max_length,
    batch_size,
    model_name,
    data_path,
    task_type,
    save_model_path,
    start_model,
    device,
    test_split_size,
    val_split_size
):
    torch.cuda.empty_cache()
    gc.collect()

    # Load the data
    genes = pd.read_csv(data_path)

    # Determine the number of labels based on the task type
    if task_type == "classification":
        n_labels = len(set(genes.Label))
    elif task_type == "regression":
        n_labels = 1
    else:
        raise ValueError(f"task_type error: {task_type}")


    # Initialize the model
    starting_point_model = FineTunedBERT(
        pool=pool,
        model_name=model_name,
        gene2vec_flag=False,
        gene2vec_hidden=200,
        task_type="unsupervised",  # Note: task_type might need to be set correctly
        n_labels=1,  # Task type is unsupervised here, but this may need adjustment
        device=device
    ).to(device)

    # Load the model state
    if start_model:
        starting_point_model.load_state_dict(torch.load(start_model))

    # Process the data
    train_loader, val_loader, test_loader = process_data(
        genes, max_length, batch_size,
        test_split_size=test_split_size,
        val_split_size=val_split_size,
        task_type=task_type,
        model_name=model_name
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Train the model
    sol_model, history, labels_test, best_pred = trainer(
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=lr,
        pool=pool,
        max_length=max_length,
        device=device,
        task_type=task_type,
        n_labels=n_labels,
        model_name=model_name,
        load_model=starting_point_model,
        save_model_path=save_model_path
    )

    return sol_model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FineTunedBERT model on gene data")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=3e-05, help="Learning rate")
    parser.add_argument("--pool", type=str, default="mean", help="Pooling strategy")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", help="Pre-trained model name")
    parser.add_argument("--data_path", type=str, default="data/combined_solubility.csv", help="Path to the dataset CSV file")
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], default="classification", help="Task type (classification or regression)")
    parser.add_argument("--save_model_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--start_model", type=str, default=None, help="Path to the starting model checkpoint")
    parser.add_argument("--test_split_size", type=float, default=0.15, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--val_split_size", type=float, default=0.15, help="Proportion of the dataset to include in the validation split")
    
    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parse_parameters(
        epochs=args.epochs,
        lr=args.lr,
        pool=args.pool,
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_name=args.model_name,
        data_path=args.data_path,
        task_type=args.task_type,
        save_model_path=args.save_model_path,
        start_model=args.start_model,
        device=device,
        test_split_size=args.test_split_size,
        val_split_size=args.val_split_size
    )
