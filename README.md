# LitGene
LitGene is a representation-learning based model that enhances gene representation using textual information from scientific literature. Contrastive learning is used to enhace gene embeddings, using similar Gene Ontology (GO) terms to create positive and negative gene pairs. This repository contains the necessary codebase, model weights, and datasets to replicate our training and evaluation of the model. Our study experiments with the use of this model on a multitude of downstream biomedical tasks:

1. Protein Soluability
2. 


Links: [[arXiv](https://www.biorxiv.org/content/10.1101/2024.08.07.606674v2.abstract)] [[Interactive Webpage](http://64.106.39.56:5000/)]

#### Outline

## Model Weights

## Usage
Pre-reqs: Ensue that Anaconda is installed
1. Clone this repository
```bash
git clone https://github.com/sposhiy33/LitGeneUpdate.git
cd LitGeneUpdate
```
2. Setup conda environment
```bash
conda env create --name LitGene --file dependencies/conda/requirements.yml
conda activate LitGene
```
### Training - Constrastive Learning for Gene Embeddings

### Downstream Tasks

### Evaluation

### Interpretability Analysis 

## Hyperparameters
- ```--epochs``` number of epochs
  
- ```--lr``` learning rate, default=3e-05
  
- ```--pool``` pooling, default="mean"
  
- ```--max_length``` maximum sequence length, default=512
  
- ```--batch_size``` training batch size, default=50
  
- ```--model_name``` specify name of pre-trained models, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  
- ```--data_path``` path to dataset (CSV), default="data/combined_solubility.csv"
  
- ```--task_type``` task type (classification or regression), default="classification"

- ```--test_split_size``` proportion of the dataset to include in the test split, default=0.15

- ```--val_split_size``` proportion of the dataset to include in the validation split, defasult=0.15

- ```--save_model_path``` specify directory at which to save the trained model checkpoint

- ```--start_model``` path to saved model checkpoint, used to resume training

## Citation
```bibtex
@article {Jararweh2024.08.07.606674,
	author = {Jararweh, Ala and Macaulay, Oladimeji and Arredondo, David and Oyebamiji, Olufunmilola M and Hu, Yue and Tafoya, Luis and Zhang, Yanfu and Virupakshappa, Kushal and Sahu, Avinash},
	title = {LitGene: a transformer-based model that uses contrastive learning to integrate textual information into gene representations},
	elocation-id = {2024.08.07.606674},
	year = {2024},
	doi = {10.1101/2024.08.07.606674},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/08/08/2024.08.07.606674},
	eprint = {https://www.biorxiv.org/content/early/2024/08/08/2024.08.07.606674.full.pdf},
	journal = {bioRxiv}
}
```
