import os
import logging
import torch
from transformers import BitsAndBytesConfig

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 路径配置 (使用绝对路径) ---
# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# pipeline/mea_kg_builder -> pipeline -> 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

# 1. 输入：本体定义文件 (OWL)
ONTOLOGY_DIR = os.path.join(PROJECT_ROOT, 'data', 'ontology')
ONTOLOGY_OWL_PATH = os.path.join(ONTOLOGY_DIR, 'MEA_Ontology.owl')

# 2. 输入：预处理后的语料库
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus_preprocessed')
ACADEMIC_DIR = os.path.join(CORPUS_DIR, 'academic')
NEWS_DIR = os.path.join(CORPUS_DIR, 'news')
WEB_DIR = os.path.join(CORPUS_DIR, 'web')

# 3. 输入：LoRA 微调权重 (来自 experiments 阶段)
# 指向 eval_llm_lora.py 训练输出的文件夹
LORA_ADAPTER_DIR = os.path.join(PROJECT_ROOT, 'data', 'experiments', 'lora_output')

# 4. 输出：提取结果
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'knowledge_graph_build')
EXTRACTION_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'extraction_results.jsonl')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 模型配置 (本地加载关键部分) ---

# 1. 基础模型路径 (必须是 Hugging Face 格式)
# 请修改为您本地 gpt-oss:20b 的实际 HF 格式文件夹路径，或者 HuggingFace ID
# 例如: "meta-llama/Meta-Llama-3-8B" 或 "C:/Models/gpt-oss-20b"
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B" 

# 2. 量化配置 (4-bit, 节省显存)
# 这必须与 eval_llm_lora.py 中的训练配置保持一致
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 3. 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 提取参数 ---
# [重要] 本地加载大模型显存占用极高，必须强制单线程 (MAX_WORKERS = 1)
# 否则多进程会重复加载模型导致 OOM (Out Of Memory)
MAX_WORKERS = 1 
# 文本分块大小
CHUNK_SIZE = 2048
