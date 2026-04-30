import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Tuple, Dict, Optional
import wandb

# ==========================================
# 1. Reproducibility
# ==========================================
def set_seeds(seed: int = 42) -> None:
    """
    Ensures total reproducibility across all computational libraries.
    This guarantees that every run yields the exact same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 2. Data Handling & PyTorch Datasets
# ==========================================
class FoodFactsDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DataHandler:
    """Manages the raw data and prepares it for specific continual learning stages."""

    def __init__(self, train_path: str, val_path: str):
        print("Loading and optimising dataset size...")
        df_train = pd.read_csv(train_path).dropna(subset=['text', 'label'])
        df_val = pd.read_csv(val_path).dropna(subset=['text', 'label'])

        # Shuffle and sample to ensure balanced, fast training
        self.df_train = df_train.sample(frac=1.0, random_state=42).groupby('label').head(1500).reset_index(drop=True)
        self.df_val = df_val.sample(frac=1.0, random_state=42).groupby('label').head(200).reset_index(drop=True)

    def get_stage_data(self, start_class: int, end_class: int) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Extracts text and labels for a specific stage, removing Pandas from downstream tasks."""
        stage_train = self.df_train[(self.df_train['label'] >= start_class) & (self.df_train['label'] < end_class)]
        stage_val = self.df_val[(self.df_val['label'] >= start_class) & (self.df_val['label'] < end_class)]

        return (
            stage_train['text'].tolist(), stage_train['label'].tolist(),
            stage_val['text'].tolist(), stage_val['label'].tolist()
        )

    def get_replay_samples(self, start_class: int, end_class: int, samples_per_class: int) -> Tuple[List[str], List[int]]:
        """Selects a subset of current stage data to save for future replay."""
        stage_df = self.df_train[(self.df_train['label'] >= start_class) & (self.df_train['label'] < end_class)]
        memory_df = stage_df.sample(frac=1.0, random_state=42).groupby('label').head(samples_per_class)
        return memory_df['text'].tolist(), memory_df['label'].tolist()

# ==========================================
# 3. The Replay Buffer (PyTorch Native)
# ==========================================
class ReplayBuffer:
    """
    Maintains memory of previous tasks natively in Python lists to prevent
    expensive Pandas concatenations during the training loop.
    """
    def __init__(self):
        self.memory_texts: List[str] = []
        self.memory_labels: List[int] = []

    def update_memory(self, texts: List[str], labels: List[int]) -> None:
        """Appends new examples to the buffer."""
        self.memory_texts.extend(texts)
        self.memory_labels.extend(labels)

    def get_memory_dataset(self, tokenizer: DistilBertTokenizer) -> Optional[FoodFactsDataset]:
        """Returns the current buffer as a PyTorch Dataset, or None if empty."""
        if not self.memory_texts:
            return None
        return FoodFactsDataset(self.memory_texts, self.memory_labels, tokenizer)

# ==========================================
# 4. The Continual Learner Core
# ==========================================
class ContinualLearner:
    """Orchestrates the training, LoRA integration, evaluation, and experiment tracking."""

    def __init__(self, num_stages: int = 10, total_classes: int = 100, memory_size_per_class: int = 50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_stages = num_stages
        self.total_classes = total_classes
        self.classes_per_stage = total_classes // num_stages
        self.memory_size_per_class = memory_size_per_class

        # Initialise a fresh Model & Tokenizer for every instance to prevent weight leakage
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        base_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=total_classes)

        # PEFT / LoRA Configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "k_lin", "v_lin"]
        )
        self.model = get_peft_model(base_model, lora_config).to(self.device)

        self.val_loaders_seen: List[DataLoader] = []
        self.accuracy_matrix: List[List[float]] = []
        self.memory_buffer = ReplayBuffer()

    def calculate_bwt(self) -> float:
        """Calculates Backward Transfer (BWT) using the accuracy matrix."""
        num_tasks = len(self.accuracy_matrix)
        if num_tasks < 2: return 0.0
        return sum((self.accuracy_matrix[-1][i] - self.accuracy_matrix[i][i]) for i in range(num_tasks - 1)) / (num_tasks - 1)

    def evaluate(self, current_stage: int) -> List[float]:
        """Evaluates the model across all tasks seen so far."""
        self.model.eval()
        accuracies = []
        active_classes = (current_stage + 1) * self.classes_per_stage

        with torch.no_grad():
            for task_id, val_loader in enumerate(self.val_loaders_seen):
                correct, total = 0, 0
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)

                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    logits[:, active_classes:] = -float('inf') # Mask unseen classes

                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                task_acc = correct / total if total > 0 else 0.0
                accuracies.append(task_acc)
                print(f"    -> Accuracy on Task {task_id}: {task_acc * 100:.2f}%")

                # Log to Weights & Biases
                wandb.log({f"eval/Task_{task_id}_Accuracy": task_acc * 100, "stage": current_stage})

        return accuracies

    def train(self, data_handler: DataHandler, use_replay: bool = False, epochs_per_stage: int = 3) -> Tuple[nn.Module, List[List[float]]]:
        """Main training loop orchestrating data mixing and parameter updates."""
        
        run_name = "Experience_Replay" if use_replay else "Naive_Baseline"
        wandb.init(project="Roseway_KTP_Continual_Learning", name=run_name, config={
            "epochs": epochs_per_stage,
            "memory_size": self.memory_size_per_class if use_replay else 0,
            "model": "DistilBERT+LoRA",
            "strategy": run_name
        })

        for stage in range(self.num_stages):
            start_class = stage * self.classes_per_stage
            end_class = (stage + 1) * self.classes_per_stage
            print(f"\n{'='*40}\nSTAGE {stage}: Classes {start_class} to {end_class - 1}\n{'='*40}")

            # 1. Fetch Data
            train_texts, train_labels, val_texts, val_labels = data_handler.get_stage_data(start_class, end_class)
            current_task_dataset = FoodFactsDataset(train_texts, train_labels, self.tokenizer)

            # 2. Mix with Replay Buffer (ONLY if use_replay is True)
            if use_replay:
                memory_dataset = self.memory_buffer.get_memory_dataset(self.tokenizer)
                if memory_dataset is not None:
                    print(f"Mixing replay samples from previous stages...")
                    combined_dataset = ConcatDataset([current_task_dataset, memory_dataset])
                else:
                    combined_dataset = current_task_dataset
            else:
                combined_dataset = current_task_dataset

            train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

            val_dataset = FoodFactsDataset(val_texts, val_labels, self.tokenizer)
            self.val_loaders_seen.append(DataLoader(val_dataset, batch_size=32, shuffle=False))

            # 3. Model Training
            optimizer = AdamW(self.model.parameters(), lr=1e-4)
            loss_fn = nn.CrossEntropyLoss()

            self.model.train()
            for epoch in range(epochs_per_stage):
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)

                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    masked_logits = logits.clone()
                    masked_logits[:, end_class:] = -float('inf')

                    loss = loss_fn(masked_logits, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"  Epoch {epoch + 1}/{epochs_per_stage} | Training Loss: {avg_loss:.4f}")
                wandb.log({"train/loss": avg_loss, "epoch_absolute": stage * epochs_per_stage + epoch})

            # 4. Update Memory Buffer (ONLY if use_replay is True)
            if use_replay:
                mem_texts, mem_labels = data_handler.get_replay_samples(start_class, end_class, self.memory_size_per_class)
                self.memory_buffer.update_memory(mem_texts, mem_labels)

            print(f"\nEvaluating Stage {stage}...")
            stage_accuracies = self.evaluate(stage)
            self.accuracy_matrix.append(stage_accuracies)
            wandb.log({"eval/avg_accuracy_seen_tasks": np.mean(stage_accuracies) * 100, "stage": stage})

        bwt = self.calculate_bwt()
        print(f"\nFinal Backward Transfer (BWT): {bwt * 100:.2f}%")
        wandb.log({"final/BWT": bwt * 100})
        wandb.finish() # Closes the W&B run

        return self.model, self.accuracy_matrix

# ==========================================
# 5. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    from IPython.display import display
    
    # 1. Enforce strict reproducibility
    set_seeds(42)

    # 2. Initialise Data
    train_csv_path = '/content/clean_train.csv' 
    val_csv_path = '/content/clean_val.csv'     

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        print("ERROR: clean_train.csv and clean_val.csv must be in the same folder as this script.")
    else:
        data_handler = DataHandler(train_csv_path, val_csv_path)
        
        # ----------------------------------------------------
        # EXPERIMENT 1: NAÏVE BASELINE
        # ----------------------------------------------------
        print("\n\n" + "#"*50)
        print("COMMENCING EXPERIMENT 1: NAÏVE BASELINE")
        print("#"*50)
        baseline_learner = ContinualLearner(num_stages=10, total_classes=100)
        _, baseline_matrix = baseline_learner.train(data_handler, use_replay=False, epochs_per_stage=3)
        
        df_baseline = pd.DataFrame(baseline_matrix)
        df_baseline.index = [f"Trained on Stage {i}" for i in range(10)]
        df_baseline.columns = [f"Eval on Stage {i}" for i in range(10)]
        print("\n=== NAÏVE BASELINE ACCURACY MATRIX ===")
        print(df_baseline.round(4) * 100)

        # ----------------------------------------------------
        # EXPERIMENT 2: EXPERIENCE REPLAY
        # ----------------------------------------------------
        print("\n\n" + "#"*50)
        print("COMMENCING EXPERIMENT 2: EXPERIENCE REPLAY")
        print("#"*50)
        # Note: A completely fresh ContinualLearner is initialised to ensure zero weight leakage
        replay_learner = ContinualLearner(num_stages=10, total_classes=100, memory_size_per_class=50)
        _, replay_matrix = replay_learner.train(data_handler, use_replay=True, epochs_per_stage=3)
        
        df_replay = pd.DataFrame(replay_matrix)
        df_replay.index = [f"Trained on Stage {i}" for i in range(10)]
        df_replay.columns = [f"Eval on Stage {i}" for i in range(10)]
        print("\n=== EXPERIENCE REPLAY ACCURACY MATRIX ===")
        print(df_replay.round(4) * 100)
