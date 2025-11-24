# mea_kg_builder/config.py
# 全局配置中心 (LoRA 微调适配版)

import os
import logging

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
# 如果您运行了 eval_llm_lora.py，适配器会保存在这里
LORA_ADAPTER_DIR = os.path.join(PROJECT_ROOT, 'data', 'experiments', 'lora_output')

# 4. 输出：提取结果
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'knowledge_graph_build')
EXTRACTION_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'extraction_results.jsonl')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 模型配置 ---

# [关键修改] 指向微调后的模型
# 注意：Ollama 无法直接加载 .bin/.safetensors 适配器文件。
# 您需要创建一个 Modelfile (FROM gpt-oss:20b -> ADAPTER /path/to/lora)，然后运行 `ollama create mea-kg-model`
OLLAMA_MODEL_NAME = "mea-kg-model" # 建议将微调后的模型命名为此名称
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_API_KEY = "ollama"

# 备用：如果没有微调，回退到基础模型
# OLLAMA_MODEL_NAME = "gpt-oss:20b"

# --- 提取参数 ---
# 本地大模型推理显存占用高，建议由 1 开始测试
MAX_WORKERS = 1 
# 文本分块大小
CHUNK_SIZE = 2048
