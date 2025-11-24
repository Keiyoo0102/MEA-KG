import os
import json
import re
import logging
import numpy as np
import torch
from typing import List
from tqdm.auto import tqdm

# Third-party libraries
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
import tenacity
from sentence_transformers import SentenceTransformer, util
from seqeval.metrics import classification_report

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Paths (Relative to 'experiments/')
BASE_DATA_DIR = '../data'
DATASET_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'datasets')
RESULT_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'results')

# Input Files
DATA_FILES = {
    'train': os.path.join(DATASET_DIR, 'train.txt'), # Used for RAG Knowledge Base
    'test': os.path.join(DATASET_DIR, 'test.txt'),   # Target for evaluation
}

# API Credentials
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")

# Model Settings
LLM_MODEL_NAME = 'gpt-4o-mini'
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_EXAMPLES = 3 # Number of few-shot examples to retrieve

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

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"--- Device: {DEVICE} ---")

# --- 1. Setup Labels & Schema ---

def create_label_maps(entity_types):
    """Creates BIO tag list."""
    labels = ["O"]
    for entity in entity_types:
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")
    return labels

LABELS_LIST = create_label_maps(ENTITY_TYPES)

# Pydantic Model for Structured Output (Instructor)
class TokenTag(BaseModel):
    token: str = Field(..., description="The original token from the sentence.")
    tag: str = Field(..., description=f"The BIO tag for this token, must be one of {LABELS_LIST}")

# --- 2. Data Loading & Processing ---

def read_conll_sentences(file_path):
    """Parses CoNLL file into list of (tokens, tags) tuples."""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sentence_blocks = re.split(r'\n\s*\n', content)
        for block in sentence_blocks:
            block = block.strip()
            if not block: continue
            
            tokens, tags = [], []
            for line in block.split('\n'):
                line = line.strip()
                if not line: continue
                
                # Robust split
                parts = line.split('\t')
                if len(parts) < 2: parts = line.split(' ')
                
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])
            
            if tokens:
                sentences.append((tokens, tags))
                
        logging.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []

def build_example_database(file_path, sbert_model):
    """
    Encodes the training dataset using SBERT to create a searchable knowledge base.
    Returns: embeddings tensor, original data list
    """
    train_sentences = read_conll_sentences(file_path)
    if not train_sentences:
        return None, None

    # Convert to plain text for embedding
    plain_text_sentences = [" ".join(tokens) for tokens, tags in train_sentences]
    
    logging.info(f"Encoding {len(plain_text_sentences)} examples for RAG...")
    db_embeddings = sbert_model.encode(
        plain_text_sentences,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=DEVICE
    )
    return db_embeddings, train_sentences

# --- 3. LLM Interaction ---

# Initialize Instructor Client
try:
    client = instructor.patch(
        OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60.0),
        mode=instructor.Mode.JSON
    )
    logging.info(f"LLM Client initialized: {LLM_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize LLM client: {e}")
    exit(1)

def build_dynamic_prompt(labels, dynamic_examples):
    """Constructs a prompt with dynamically retrieved few-shot examples."""
    prompt = f"""You are an expert Named Entity Recognition (NER) system.
Your task is to label entities in a given sentence.
The available entity labels are:
{labels}

You will be given a sentence as a JSON list of tokens.
Your task is to return a list of JSON objects, one for each token, matching the requested format.
You MUST return exactly one JSON object for each token.
"""
    for i, (ex_tokens, ex_tags) in enumerate(dynamic_examples):
        user_example = str(ex_tokens)
        assistant_example_list = [{"token": t, "tag": g} for t, g in zip(ex_tokens, ex_tags)]
        assistant_example = json.dumps(assistant_example_list, ensure_ascii=False)
        
        prompt += f"\n---\nEXAMPLE {i + 1}:\nUser:\n{user_example}\n\nAssistant:\n{assistant_example}\n---"
    
    return prompt

# Retry Logic using Tenacity
retry_policy = tenacity.retry(
    stop=(tenacity.stop_after_delay(120) | tenacity.stop_after_attempt(5)),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    reraise=True
)

@retry_policy
def get_llm_prediction(tokens, true_tags, system_prompt):
    """Calls LLM with retry logic and schema enforcement."""
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        response_model=List[TokenTag],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(tokens)}
        ],
        temperature=0.0 # Deterministic
    )
    
    pred_tags = [item.tag for item in response]
    
    # Validate length alignment
    if len(pred_tags) != len(true_tags):
        raise ValueError(f"Length mismatch: Expected {len(true_tags)}, got {len(pred_tags)}")
        
    return pred_tags

# --- 4. Evaluation Loop ---

def evaluate_llm_rag(test_sentences, sbert_model, db_embeddings, db_data):
    all_true, all_pred = [], []
    
    logging.info(f"Starting RAG Evaluation on {len(test_sentences)} sentences...")
    
    for tokens, true_tags in tqdm(test_sentences, desc="LLM-RAG Eval"):
        try:
            # 1. Retrieval
            query_text = " ".join(tokens)
            query_embedding = sbert_model.encode(query_text, convert_to_tensor=True, device=DEVICE)
            
            # Semantic Search
            hits = util.pytorch_cos_sim(query_embedding, db_embeddings)[0]
            top_k_indices = torch.topk(hits, k=TOP_K_EXAMPLES).indices.cpu().tolist()
            dynamic_examples = [db_data[i] for i in top_k_indices]
            
            # 2. Prompt Construction
            system_prompt = build_dynamic_prompt(LABELS_LIST, dynamic_examples)
            
            # 3. Inference
            pred_tags = get_llm_prediction(tokens, true_tags, system_prompt)
            
            all_true.append(true_tags)
            all_pred.append(pred_tags)
            
        except Exception as e:
            logging.error(f"Failed to process sentence: {tokens[:5]}... Error: {e}")
            continue

    if not all_true:
        logging.error("No sentences processed successfully.")
        return None

    report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
    return report["micro avg"]["f1-score"], report, (all_true, all_pred)

# --- 5. Main Execution ---

def main():
    # 1. Load Data
    if not os.path.exists(DATA_FILES['test']) or not os.path.exists(DATA_FILES['train']):
        logging.error(f"Data files missing in {DATASET_DIR}")
        return

    test_sentences = read_conll_sentences(DATA_FILES['test'])
    
    # 2. Initialize SBERT & Knowledge Base
    logging.info(f"Loading SBERT: {SBERT_MODEL_NAME}")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
    
    logging.info("Building RAG Knowledge Base...")
    db_embeddings, db_data = build_example_database(DATA_FILES['train'], sbert_model)
    
    if db_embeddings is None:
        logging.error("Failed to build knowledge base.")
        return

    # 3. Run Evaluation
    result = evaluate_llm_rag(test_sentences, sbert_model, db_embeddings, db_data)
    
    if result:
        f1, report_dict, _ = result
        
        print("\n" + "=" * 50)
        print(f"  LLM ({LLM_MODEL_NAME}) + RAG Evaluation Report")
        print("=" * 50)
        print(f"Precision: {report_dict['micro avg']['precision']:.4f}")
        print(f"Recall:    {report_dict['micro avg']['recall']:.4f}")
        print(f"F1-Score:  {report_dict['micro avg']['f1-score']:.4f}")
        print("=" * 50)
        
        # Optional: Save detailed report
        os.makedirs(RESULT_DIR, exist_ok=True)
        # You can add logic to save CSV report here if needed

if __name__ == '__main__':
    main()
