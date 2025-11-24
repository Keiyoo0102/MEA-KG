import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from collections import defaultdict

# Third-party libraries
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
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_lstm_crf_model.pth')
HISTORY_CSV_PATH = os.path.join(RESULT_DIR, 'lstm_crf_training_history.csv')

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

# Model Hyperparameters
EMBEDDING_DIM = 100
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 2
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3  # Higher LR for non-pretrained models

# Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"--- Device: {DEVICE} ---")

# --- 1. Setup Labels & Vocabulary ---

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

def build_vocab(file_path, min_freq=1):
    """Builds token_to_id mapping from training data."""
    token_counts = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sentence_blocks = re.split(r'\n\s*\n', content)

    for block in sentence_blocks:
        block = block.strip()
        if not block: continue
        
        for line in block.split('\n'):
            line = line.strip()
            if not line: continue

            parts = line.split('\t')
            if len(parts) < 2: parts = line.split(' ')
            
            if len(parts) >= 2:
                token = parts[0]
                token_counts[token] += 1

    # 0 = <PAD>, 1 = <UNK>
    token_to_id = {'<PAD>': 0, '<UNK>': 1}
    for token, count in token_counts.items():
        if count >= min_freq:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    logging.info(f"Vocabulary Size: {len(token_to_id)}")
    return token_to_id

# Build Vocab (Only from Train set)
if not os.path.exists(DATA_FILES['train']):
    logging.error(f"Training data not found at {DATA_FILES['train']}")
    exit(1)

TOKEN_TO_ID = build_vocab(DATA_FILES['train'])
VOCAB_SIZE = len(TOKEN_TO_ID)

# --- 2. Data Loading & Preprocessing ---

class NerDataset(Dataset):
    def __init__(self, file_path, token_to_id, label_to_id):
        self.token_ids_list = []
        self.tag_ids_list = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        sentence_blocks = re.split(r'\n\s*\n', content)

        for block in sentence_blocks:
            block = block.strip()
            if not block: continue

            sentence_token_ids = []
            sentence_tag_ids = []

            for line in block.split('\n'):
                line = line.strip()
                if not line: continue

                parts = line.split('\t')
                if len(parts) < 2: parts = line.split(' ')
                
                if len(parts) >= 2:
                    token, tag = parts[0], parts[-1]
                    
                    # Map to IDs
                    sentence_token_ids.append(token_to_id.get(token, token_to_id['<UNK>']))
                    sentence_tag_ids.append(label_to_id.get(tag, label_to_id['O']))

            if sentence_token_ids:
                self.token_ids_list.append(sentence_token_ids)
                self.tag_ids_list.append(sentence_tag_ids)

    def __len__(self):
        return len(self.token_ids_list)

    def __getitem__(self, idx):
        return self.token_ids_list[idx], self.tag_ids_list[idx]

def collate_fn(batch):
    """Pads sequences to the longest in the batch."""
    PAD_TOKEN_ID = TOKEN_TO_ID['<PAD>']
    PAD_LABEL_ID = -100

    token_ids_list, label_ids_list = zip(*batch)
    max_len = max(len(ids) for ids in token_ids_list)

    padded_token_ids = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    padded_label_ids = torch.full((len(batch), max_len), PAD_LABEL_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i in range(len(batch)):
        seq_len = len(token_ids_list[i])
        padded_token_ids[i, :seq_len] = torch.tensor(token_ids_list[i], dtype=torch.long)
        padded_label_ids[i, :seq_len] = torch.tensor(label_ids_list[i], dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    return {
        'input_ids': padded_token_ids,
        'labels': padded_label_ids,
        'attention_mask': attention_mask
    }

# Setup Directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load Datasets
train_dataset = NerDataset(DATA_FILES['train'], TOKEN_TO_ID, LABEL_TO_ID)
val_dataset = NerDataset(DATA_FILES['validation'], TOKEN_TO_ID, LABEL_TO_ID)
test_dataset = NerDataset(DATA_FILES['test'], TOKEN_TO_ID, LABEL_TO_ID)

# Create Loaders
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)

# --- 3. Model Architecture (LSTM + CRF) ---

class LstmCrfForNer(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim, hidden_dim, pad_token_id):
        super(LstmCrfForNer, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id
        )
        self.dropout = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        
        # Input to linear is hidden_dim * 2 (Bidirectional)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Embed
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # 2. LSTM
        lstm_output, _ = self.lstm(embeddings)
        lstm_output = self.dropout(lstm_output)
        
        # 3. Linear
        emissions = self.classifier(lstm_output)
        
        # 4. CRF
        mask = attention_mask.bool()
        
        if labels is not None:
            cleaned_labels = labels.clone()
            cleaned_labels[cleaned_labels == -100] = 0
            loss = -self.crf(emissions, cleaned_labels, mask=mask, reduction='mean')
            return loss, emissions
        else:
            decoded_tags = self.crf.decode(emissions, mask=mask)
            return None, decoded_tags

# Initialize Model
model = LstmCrfForNer(
    vocab_size=VOCAB_SIZE,
    num_labels=len(LABELS_LIST),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=LSTM_HIDDEN_DIM,
    pad_token_id=TOKEN_TO_ID['<PAD>']
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. Training & Evaluation Functions ---

def evaluate(model_to_eval, dataloader):
    model_to_eval.eval()
    all_true, all_pred = [], []

    for batch in dataloader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            _, predicted_ids = model_to_eval(**batch)

        for i, true_ids in enumerate(labels.cpu().numpy()):
            pred_ids = predicted_ids[i]
            true_str, pred_str = [], []
            
            for j, t_id in enumerate(true_ids):
                if t_id != -100:
                    true_str.append(ID_TO_LABEL[t_id])
                    # Safety check for length
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
        
        # Loss
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Training Loss', fontsize=14, color='black')
        ax1.plot(history_df['epoch'], history_df['train_loss'], 
                 color=color_loss, marker='o', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Metrics
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
    logging.info("--- Starting Training ---")
    best_f1 = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Optimization
            optimizer.zero_grad()
            loss, _ = model(**batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        
        # Validation
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
    plot_training_history(df, "LSTM-CRF")
    return df

def main():
    if os.path.exists(HISTORY_CSV_PATH):
        logging.info(f"Found existing history: {HISTORY_CSV_PATH}. Loading...")
        try:
            df = pd.read_csv(HISTORY_CSV_PATH)
            plot_training_history(df, "LSTM-CRF")
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
