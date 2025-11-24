import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

# Hugging Face Libraries
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    DataCollatorForTokenClassification, 
    get_scheduler
)
from datasets import Dataset, DatasetDict
from torchcrf import CRF
from seqeval.metrics import classification_report

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Paths (Relative to 'experiments/')
BASE_DATA_DIR = '../data'
DATASET_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'datasets')
MODEL_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'models')
RESULT_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'results')
FIG_DIR = os.path.join(RESULT_DIR, 'figures')

# Input Files
DATA_FILES = {
    'train': os.path.join(DATASET_DIR, 'train.txt'),
    'validation': os.path.join(DATASET_DIR, 'dev.txt'),
    'test': os.path.join(DATASET_DIR, 'test.txt'),
}

# Output Files
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_roberta_crf_model.pth')
HISTORY_CSV_PATH = os.path.join(RESULT_DIR, 'roberta_crf_training_history.csv')

# Entity Types
ENTITY_TYPES = [
    'GeologicFeature',
    'LocationRegion',
    'MissionSpacecraft',
    'Instrument',
    'MaterialComposition',
    'Process',
    'TemporalEntity',
    'Organization',
    'AstrobiologyConcept'
]

# Model Settings
# Using RoBERTa base from Hugging Face Hub
MODEL_CHECKPOINT = "roberta-base"

# Training Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5

# Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"--- Device: {DEVICE} ---")

# --- 1. Setup Labels & Tokenizer ---

def create_label_maps(entity_types):
    """Creates BIO tag mappings."""
    labels = ["O"]
    for entity in entity_types:
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    logging.info(f"Label Space Size: {len(labels)}")
    return labels, label2id, id2label

LABELS_LIST, LABEL_TO_ID, ID_TO_LABEL = create_label_maps(ENTITY_TYPES)

try:
    # RoBERTa tokenizer often needs add_prefix_space=True for pre-tokenized data
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
    logging.info(f"Tokenizer loaded: {MODEL_CHECKPOINT}")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

# --- 2. Data Loading & Preprocessing ---

def load_conll_file(file_path):
    """Parses CoNLL formatted file into a HuggingFace Dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    all_tokens = []
    all_tags = []
    sentence_blocks = re.split(r'\n\s*\n', content)

    for block in sentence_blocks:
        block = block.strip()
        if not block: continue

        tokens = []
        tags = []
        for line in block.split('\n'):
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split(' ')
            
            if len(parts) >= 2:
                token, tag = parts[0], parts[-1]
                tokens.append(token)
                tags.append(LABEL_TO_ID.get(tag, LABEL_TO_ID['O']))

        if tokens:
            all_tokens.append(tokens)
            all_tags.append(tags)

    return Dataset.from_dict({'tokens': all_tokens, 'ner_tags': all_tags})

def tokenize_and_align_labels(examples):
    """Tokenizes inputs and aligns BIO labels with sub-tokens."""
    tokenized_inputs = TOKENIZER(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Setup Directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load Data
try:
    raw_datasets = DatasetDict({
        'train': load_conll_file(DATA_FILES['train']),
        'validation': load_conll_file(DATA_FILES['validation']),
        'test': load_conll_file(DATA_FILES['test']),
    })
    
    if len(raw_datasets['train']) == 0:
        raise ValueError("Training set is empty.")
        
    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])
    
except Exception as e:
    logging.error(f"Data loading failed: {e}")
    exit(1)

# Data Loaders
data_collator = DataCollatorForTokenClassification(tokenizer=TOKENIZER)
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=BATCH_SIZE)

# --- 3. Model Architecture (RoBERTa + CRF) ---

class RobertaCrfForNer(nn.Module):
    def __init__(self, num_labels, model_name=MODEL_CHECKPOINT):
        super(RobertaCrfForNer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        # RoBERTa does not use token_type_ids
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        
        mask = attention_mask.bool()
        
        if labels is not None:
            cleaned_labels = labels.clone()
            cleaned_labels[cleaned_labels == -100] = 0 
            loss = -self.crf(emissions, cleaned_labels, mask=mask, reduction='mean')
            return loss
        else:
            decoded_tags = self.crf.decode(emissions, mask=mask)
            return decoded_tags

# Initialize Model
model = RobertaCrfForNer(num_labels=len(LABELS_LIST)).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. Training & Evaluation Functions ---

def evaluate(model_to_eval, dataloader):
    model_to_eval.eval()
    all_true, all_pred = [], []

    for batch in dataloader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            preds = model_to_eval(**batch)

        for i, pred_ids in enumerate(preds):
            true_ids = labels[i].cpu().numpy()
            true_str, pred_str = [], []
            
            for j, t_id in enumerate(true_ids):
                if t_id != -100:
                    true_str.append(ID_TO_LABEL[t_id])
                    if j < len(pred_ids):
                        pred_str.append(ID_TO_LABEL[pred_ids[j]])
                    else:
                        pred_str.append("O")
            
            all_true.append(true_str)
            all_pred.append(pred_str)

    report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
    return report["micro avg"]["f1-score"], report, (all_true, all_pred)

def plot_training_history(history_df, model_name):
    """Generates a publication-quality dual-axis plot."""
    try:
        sns.set_theme(style="ticks")
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        plt.title(model_name, fontsize=16, pad=20)
        
        # Loss (Left Axis)
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Training Loss', fontsize=14, color='black')
        ax1.plot(history_df['epoch'], history_df['train_loss'], 
                 color=color_loss, marker='o', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Metrics (Right Axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Score', fontsize=14, color='black')
        
        ax2.plot(history_df['epoch'], history_df['val_f1'], 
                 color='tab:red', marker='s', linewidth=2.5, label='Validation F1')
        ax2.plot(history_df['epoch'], history_df['val_precision'], 
                 color='tab:green', marker='^', linestyle='--', label='Precision')
        ax2.plot(history_df['epoch'], history_df['val_recall'], 
                 color='tab:orange', marker='x', linestyle=':', label='Recall')
        
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0.0, 1.05)

        sns.despine(ax=ax1, top=True, right=True)
        sns.despine(ax=ax2, top=True, left=True, bottom=True, right=False)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, 
                   loc='center right', bbox_to_anchor=(0.95, 0.5), 
                   fancybox=True, shadow=True, title="Metrics")

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, f"{model_name}_curves.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logging.info(f"Plot saved to {fig_path}")
    except Exception as e:
        logging.error(f"Plotting error: {e}")

# --- 5. Main Training Loop ---

def train():
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    logging.info("--- Starting Training ---")
    best_f1 = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        
        val_f1, val_report, _ = evaluate(model, eval_dataloader)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_f1': val_f1,
            'val_precision': val_report['micro avg']['precision'],
            'val_recall': val_report['micro avg']['recall']
        })
        
        logging.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logging.info(f"New best model saved to {BEST_MODEL_PATH}")

    # Save History
    df = pd.DataFrame(history)
    df.to_csv(HISTORY_CSV_PATH, index=False)
    plot_training_history(df, "RoBERTa-CRF")
    return df

def main():
    if os.path.exists(HISTORY_CSV_PATH):
        logging.info(f"Found existing history: {HISTORY_CSV_PATH}. Loading...")
        try:
            df = pd.read_csv(HISTORY_CSV_PATH)
            plot_training_history(df, "RoBERTa-CRF")
        except:
            logging.warning("Failed to read history, restarting training.")
            train()
    else:
        train()

    logging.info("--- Final Evaluation on Test Set ---")
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    except:
        logging.warning("Could not load best model, using current weights.")
    
    test_f1, _, (y_true, y_pred) = evaluate(model, test_dataloader)
    print("\n" + classification_report(y_true, y_pred, digits=4))
    logging.info(f"Final Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
