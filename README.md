# Roseway KTP: Continual Learning for Food Category Classification

**Candidate:** Dhruv Patel  
**Role:** AI-Assisted Formulation Lead (KTP Associate)

## Project Overview
This repository contains the codebase for a Class-Incremental Learning (CIL) pipeline. The objective of this project is to sequentially train a Transformer-based model (`distilbert-base-uncased`) to classify 100 food product categories across 10 incremental stages without suffering from **Catastrophic Forgetting**.

To adhere to strict computational efficiency and memory constraints, the architecture utilises **Parameter-Efficient Fine-Tuning (PEFT)** via Low-Rank Adaptation (LoRA), alongside a targeted **Experience Replay** memory buffer to protect historical knowledge.

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
