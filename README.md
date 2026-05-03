# Roseway KTP: Continual Learning for Food Category Classification

**Candidate:** Dhruv Patel  
**Role:** AI-Assisted Formulation Lead (KTP Associate)

## Project Overview
This repository contains the codebase for a Class-Incremental Learning (CIL) pipeline. The objective of this project is to sequentially train a Transformer-based model (`distilbert-base-uncased`) to classify 100 food product categories across 10 incremental stages without suffering from **Catastrophic Forgetting**.

To adhere to strict computational efficiency and memory constraints, the architecture utilises **Parameter-Efficient Fine-Tuning (PEFT)** via Low-Rank Adaptation (LoRA), alongside a targeted **Experience Replay** memory buffer to protect historical knowledge.

## Experiment Registration & Reproducibility

To ensure full transparency, scientific rigour, and reproducibility of the continual learning methodologies evaluated in this repository, the experimental design and baseline protocols have been registered on the Open Science Framework (OSF).

* **OSF Registration DOI:** [10.17605/OSF.IO/XQGDY](https://doi.org/10.17605/OSF.IO/XQGDY)

This registration formally documents the class-incremental learning strategy, evaluation metrics (including quantitative measures of catastrophic forgetting), and model architecture choices for the Open Food Facts classification task.

## Repository Structure
* `continual_learning.py`: The main execution script containing the OOP-structured Continual Learner, custom PyTorch Datasets, and Replay Buffer logic.
* `requirements.txt`: Required dependencies for execution.
* `Technical_Summary_Report.pdf`: A 2-page critical analysis of the methodology, architecture, and resulting metrics.
* `Future_Architecture_Proposal.pdf`: A theoretical framework for utilising shared ingredients via Semantic Anchoring to further improve classification.

## Data Preparation
The code expects two cleaned CSV files derived from the Open Food Facts dataset. Place these in the root directory of the project alongside the python script:
* `clean_train.csv`
* `clean_val.csv`

*(Note: Due to file size limits, the raw CSV datasets are not included in this repository).*

## Setup and Installation
It is recommended to run this project within a Python Virtual Environment (e.g., Conda or venv). 

1. Clone the repository and navigate to the directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Execution & Weights and Biases (W&B)

This project utilises [Weights & Biases (W&B)](https://wandb.ai/) for rigorous experiment tracking and metric logging.

**Reviewer Instructions:** 

The raw Open Food Facts dataset was filtered for the top 100 classes and capped at 1,500 samples per class to generate the training data. The data cleaning script is available as a .zip.

To run this code locally without authenticating a personal W&B account or requiring an API key, simply prepend your execution command with the offline flag. This will disable cloud syncing and print all progress and metrics directly to your local console:
```bash
WANDB_MODE=offline python continual_learning.py
