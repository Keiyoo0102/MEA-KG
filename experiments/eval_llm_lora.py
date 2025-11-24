import os
import re
import json
import logging
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from seqeval.metrics import classification_report

# --- Advanced DL imports ---
# NOTE: You need to install these specific libraries for LoRA:
# pip install transformers peft bitsandbytes trl accelerate
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig,
        TrainingArguments,
        pipeline
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer
except ImportError:
    raise ImportError("Please install required libraries: pip install transformers peft bitsandbytes trl accelerate")

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Paths
BASE_DATA_DIR = '../data'
DATASET_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'datasets')
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'lora_output')
RESULT_DIR = os.path.join(BASE_DATA_DIR, 'experiments', 'results')

DATA_FILES = {
    'train': os.path.join(DATASET_DIR, 'train.txt'),
    'test': os.path.join(DATASET_DIR, 'test.txt'),
}

# Model Settings
# Replace this with the HuggingFace path to your 'gpt-oss:20b' base model
# Example placeholders: "meta-llama/Meta-Llama-3-8B", "Qwen/Qwen1.5-14B", etc.
# Since 'gpt-oss:20b' is likely a local name, ensure the path is correct.
MODEL_NAME = "meta-llama/Meta-Llama-3-8B" # <--- CHANGE THIS to your model path

# LoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
BATCH_SIZE = 4  # Adjust based on VRAM (4 for 24GB VRAM with 7B model)
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Data Loading & Formatting ---

def read_conll_sentences(file_path):
    """Reads CoNLL file into (tokens, tags) tuples."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for block in re.split(r'\n\s*\n', content):
        block = block.strip()
        if not block: continue
        tokens, tags = [], []
        for line in block.split('\n'):
            parts = line.strip().split('\t')
            if len(parts) < 2: parts = line.strip().split(' ')
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
        if tokens: sentences.append((tokens, tags))
    return sentences

def format_instruction(tokens, tags):
    """
    Converts BIO tags into a JSON instruction format for the LLM.
    Input: "Gale Crater is..."
    Output: JSON string of entities.
    """
    sentence = " ".join(tokens)
    
    # Extract entities from BIO tags
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity: entities.append(current_entity)
            current_entity = {"token": token, "label": tag[2:]}
        elif tag.startswith("I-") and current_entity:
            current_entity["token"] += " " + token
        else:
            if current_entity: 
                entities.append(current_entity)
                current_entity = None
    if current_entity: entities.append(current_entity)
    
    # Construct JSON output
    output_json = json.dumps(entities, ensure_ascii=False)
    
    # Construct Prompt (Alpaca/Instruct style)
    prompt = f"""### Instruction:
Extract named entities from the text below. Return the result as a JSON list of objects with 'token' and 'label' keys.

### Input:
{sentence}

### Response:
{output_json}"""
    
    return {"text": prompt, "sentence": sentence, "json_label": output_json}

# --- 2. Model Setup (QLoRA) ---

def setup_model_for_training():
    logging.info(f"Loading base model: {MODEL_NAME} with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer, peft_config

# --- 3. Training Loop ---

def train_lora():
    # Load Data
    logging.info("Preparing training data...")
    train_data_raw = read_conll_sentences(DATA_FILES['train'])
    
    # Format for SFT
    formatted_data = [format_instruction(t, l) for t, l in train_data_raw]
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # Setup Model
    model, tokenizer, peft_config = setup_model_for_training()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    logging.info("Starting LoRA Fine-tuning...")
    trainer.train()
    
    logging.info(f"Saving adapter model to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Return tokenizer for immediate evaluation
    return tokenizer

# --- 4. Inference & Evaluation ---

def parse_llm_output_to_bio(text_output, original_tokens):
    """
    Heuristic parser to convert LLM JSON output back to BIO tags for evaluation.
    """
    # Extract JSON part
    try:
        json_str = text_output.split("### Response:")[-1].strip()
        entities = json.loads(json_str)
    except:
        return ['O'] * len(original_tokens) # Fallback if JSON is broken
    
    bio_tags = ['O'] * len(original_tokens)
    
    # Map entities back to tokens (Simplified Exact Match)
    token_str_list = [str(t) for t in original_tokens]
    
    for ent in entities:
        entity_text = ent.get('token', '').split()
        label = ent.get('label', 'Entity')
        
        if not entity_text: continue
        
        # Find entity sequence in original tokens
        length = len(entity_text)
        for i in range(len(token_str_list) - length + 1):
            if token_str_list[i:i+length] == entity_text:
                bio_tags[i] = f"B-{label}"
                for j in range(1, length):
                    bio_tags[i+j] = f"I-{label}"
                break # Match first occurrence
                
    return bio_tags

def evaluate_lora(tokenizer):
    logging.info("Starting Evaluation...")
    
    # Load Adapter Model for Inference
    # Reload base model to clear training memory
    del tokenizer
    torch.cuda.empty_cache()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    
    test_data = read_conll_sentences(DATA_FILES['test'])
    all_true, all_pred = [], []
    
    for tokens, true_tags in tqdm(test_data, desc="Eval LoRA"):
        sentence = " ".join(tokens)
        prompt = f"### Instruction:\nExtract named entities from the text below. Return the result as a JSON list of objects with 'token' and 'label' keys.\n\n### Input:\n{sentence}\n\n### Response:\n"
        
        try:
            result = pipe(prompt, do_sample=False)[0]['generated_text']
            pred_tags = parse_llm_output_to_bio(result, tokens)
        except Exception as e:
            logging.error(f"Inference error: {e}")
            pred_tags = ['O'] * len(tokens)
            
        all_true.append(true_tags)
        all_pred.append(pred_tags)
        
    # Metrics
    report = classification_report(all_true, all_pred, digits=4)
    print("\n" + report)
    
    # Save Results
    report_dict = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    logging.info(f"LoRA Evaluation F1: {report_dict['micro avg']['f1-score']:.4f}")

# --- Main ---

def main():
    if not torch.cuda.is_available():
        logging.error("CUDA not found. LoRA fine-tuning requires a GPU.")
        return

    # 1. Train
    if not os.path.exists(OUTPUT_DIR):
        tokenizer = train_lora()
    else:
        logging.info(f"Found existing adapter at {OUTPUT_DIR}. Skipping training.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Evaluate
    evaluate_lora(tokenizer)

if __name__ == "__main__":
    main()
