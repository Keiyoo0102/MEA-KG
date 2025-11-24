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
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig,
        TrainingArguments,
        pipeline
    )
    from peft import (
        LoraConfig, 
        get_peft_model, 
        PeftModel, 
        prepare_model_for_kbit_training,
        TaskType
    )
    from trl import SFTTrainer
except ImportError:
    raise ImportError("缺少必要的库。请运行: pip install transformers peft bitsandbytes trl accelerate")

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

# ==========================================
# 🔴 关键配置：模型路径与 LoRA 参数
# ==========================================

# 1. 基础模型路径
# 注意: transformers 无法直接加载 Ollama 的 "gpt-oss:20b" 标签。
# 您必须提供该模型权重的本地文件夹路径，或者 HuggingFace Hub 上的 ID。
# 如果您已经在本地下载了权重，请将其改为绝对路径，例如: "C:/Models/gpt-oss-20b"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # <--- 请在此处修改为您的实际模型路径

# 2. QLoRA 量化配置 (4-bit)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 3. LoRA 微调参数
LORA_CONFIG = LoraConfig(
    r=16,                    # LoRA 秩 (Rank): 越大参数越多，但也越容易过拟合
    lora_alpha=32,           # LoRA 缩放系数: 通常设为 r 的 2 倍
    lora_dropout=0.05,       # Dropout 概率
    bias="none",             # 是否训练偏置项
    task_type=TaskType.CAUSAL_LM, 
    # 目标模块: 根据您的模型架构调整。
    # Llama/Mistral/Qwen 通常包括这些:
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
)

# 4. 训练超参数
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=4,   # 根据显存调整 (24G显存可设为 4-8)
    gradient_accumulation_steps=4,   # 梯度累积，模拟大 Batch Size
    learning_rate=2e-4,              # QLoRA 常用学习率
    fp16=True,                       # 使用混合精度
    logging_steps=10,
    save_strategy="epoch",           # 每个 Epoch 保存一次
    optim="paged_adamw_32bit",       # 节省显存的优化器
    report_to="none"                 # 不上传到 WandB
)

MAX_SEQ_LENGTH = 512                 # 序列最大长度

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Data Loading & Formatting ---

def read_conll_sentences(file_path):
    """Reads CoNLL file into (tokens, tags) tuples."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
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

def format_instruction(sample):
    """
    Format data for SFTTrainer.
    Convert BIO tokens/tags into a clear instruction-response pair.
    """
    tokens = sample['tokens']
    tags = sample['tags']
    sentence = " ".join(tokens)
    
    # Extract entities from BIO tags for the "Answer" part
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
    
    # Construct JSON output string
    output_json = json.dumps(entities, ensure_ascii=False)
    
    # Construct Prompt (Alpaca style)
    text = f"""### Instruction:
Extract named entities from the text below. Return the result as a JSON list of objects with 'token' and 'label' keys.

### Input:
{sentence}

### Response:
{output_json}<|endoftext|>"""
    
    return {"text": text}

# --- 2. Training Loop ---

def train_lora():
    logging.info("--- Step 1: Preparing Data ---")
    train_data_raw = read_conll_sentences(DATA_FILES['train'])
    
    # Convert list of tuples to list of dicts for Dataset
    raw_dataset = Dataset.from_list(
        [{"tokens": t, "tags": l} for t, l in train_data_raw]
    )
    
    # Formatting is handled by SFTTrainer using 'dataset_text_field' or 'formatting_func'
    # Here we pre-format for clarity
    dataset = raw_dataset.map(format_instruction)
    logging.info(f"Training data size: {len(dataset)}")

    logging.info(f"--- Step 2: Loading Base Model ({BASE_MODEL_ID}) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix for fp16 training

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True
        )
        
        # [关键步骤] 准备模型进行 k-bit 训练 (冻结原参数，开启梯度检查点)
        model = prepare_model_for_kbit_training(model)
        
        # 应用 LoRA 配置
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()
        
    except Exception as e:
        logging.error(f"Failed to load model. Please check BASE_MODEL_ID. Error: {e}")
        return None

    logging.info("--- Step 3: Starting SFT Trainer ---")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text", # Uses the 'text' column we created
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=TRAINING_ARGS,
        packing=False,
    )
    
    trainer.train()
    
    logging.info(f"--- Step 4: Saving Adapter to {OUTPUT_DIR} ---")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Clear memory
    del model, trainer
    torch.cuda.empty_cache()
    
    return tokenizer

# --- 3. Inference & Evaluation ---

def parse_llm_output_to_bio(text_output, original_tokens):
    """Heuristic parser to convert LLM JSON output back to BIO tags."""
    try:
        # Extract part after "### Response:"
        if "### Response:" in text_output:
            response_part = text_output.split("### Response:")[-1]
            # Stop at end token if present
            response_part = response_part.split("<|endoftext|>")[0].strip()
            entities = json.loads(response_part)
        else:
            entities = []
    except:
        return ['O'] * len(original_tokens)
    
    bio_tags = ['O'] * len(original_tokens)
    token_str_list = [str(t) for t in original_tokens]
    
    if isinstance(entities, list):
        for ent in entities:
            if not isinstance(ent, dict): continue
            entity_text = ent.get('token', '').split()
            label = ent.get('label', 'Entity')
            
            if not entity_text: continue
            
            length = len(entity_text)
            # Simple greedy matching
            for i in range(len(token_str_list) - length + 1):
                if token_str_list[i:i+length] == entity_text:
                    bio_tags[i] = f"B-{label}"
                    for j in range(1, length):
                        bio_tags[i+j] = f"I-{label}"
                    break 
    return bio_tags

def evaluate_lora():
    logging.info("--- Step 5: Loading Model for Evaluation ---")
    
    # Load Base Model again
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA Adapter
    logging.info(f"Loading adapter from: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Pipeline for inference
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    
    test_data = read_conll_sentences(DATA_FILES['test'])
    all_true, all_pred = [], []
    
    logging.info(f"Evaluating on {len(test_data)} test sentences...")
    for tokens, true_tags in tqdm(test_data, desc="Eval LoRA"):
        sentence = " ".join(tokens)
        prompt = f"### Instruction:\nExtract named entities from the text below. Return the result as a JSON list of objects with 'token' and 'label' keys.\n\n### Input:\n{sentence}\n\n### Response:\n"
        
        try:
            # Inference
            result = pipe(prompt, do_sample=False, return_full_text=True)[0]['generated_text']
            pred_tags = parse_llm_output_to_bio(result, tokens)
        except Exception as e:
            logging.error(f"Inference error: {e}")
            pred_tags = ['O'] * len(tokens)
            
        all_true.append(true_tags)
        all_pred.append(pred_tags)
        
    # Metrics
    print("\n" + "="*50)
    print("  LoRA Fine-tuned Model Evaluation Report")
    print("="*50)
    print(classification_report(all_true, all_pred, digits=4))
    
    # Save Results
    report_dict = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
    os.makedirs(RESULT_DIR, exist_ok=True)
    logging.info(f"Final F1: {report_dict['micro avg']['f1-score']:.4f}")

# --- Main ---

def main():
    if not torch.cuda.is_available():
        logging.error("CUDA not found. LoRA fine-tuning requires a GPU.")
        return

    # 1. Check if adapter exists
    if not os.path.exists(OUTPUT_DIR):
        logging.info("Adapter not found. Starting training...")
        tokenizer = train_lora()
        if tokenizer is None: return # Exit if training setup failed
    else:
        logging.info(f"Found existing adapter at {OUTPUT_DIR}. Skipping training.")

    # 2. Evaluate
    evaluate_lora()

if __name__ == "__main__":
    main()
