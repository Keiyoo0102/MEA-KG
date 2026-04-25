"""
config.py — Global Configuration for MEA-KG Builder Pipeline

All sensitive credentials (API keys, passwords) are loaded exclusively from
environment variables.  Copy ``.env.example`` to ``.env``, fill in your values,
and run ``python-dotenv`` to load them automatically.

Usage
-----
    from pipeline.mea_kg_builder.config import OLLAMA_MODEL_NAME, OUTPUT_DIR
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Optionally load a local .env file (safe no-op if the file is absent or
# python-dotenv is not installed).
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Walk up from this file to find the project root's .env
    _env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv is optional; rely on shell environment variables

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
# Resolve project root: pipeline/mea_kg_builder/config.py → 3 levels up
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# Input: OWL ontology file
ONTOLOGY_DIR: Path = PROJECT_ROOT / "data" / "ontology"
ONTOLOGY_OWL_PATH: Path = ONTOLOGY_DIR / "MEA_Ontology.owl"

# Input: Pre-processed corpus directories
CORPUS_DIR: Path = PROJECT_ROOT / "data" / "corpus_preprocessed"
ACADEMIC_DIR: Path = CORPUS_DIR / "academic"
NEWS_DIR: Path = CORPUS_DIR / "news"
WEB_DIR: Path = CORPUS_DIR / "web"

# Input: QLoRA / LoRA adapter weights produced by experiments/finetune_qlora.py
LORA_ADAPTER_DIR: Path = PROJECT_ROOT / "data" / "experiments" / "lora_output"

# Output: Extraction results
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "knowledge_graph_build"
EXTRACTION_RESULTS_PATH: Path = OUTPUT_DIR / "extraction_results.jsonl"

# Ensure the output directory exists at import time
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM / Ollama Configuration  (loaded from environment variables)
# ---------------------------------------------------------------------------
OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "mea-kg-model")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
# Ollama uses a fixed dummy key; never hardcode real keys here
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "ollama")

# Cloud LLM (OpenAI-compatible)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    logger.warning(
        "OPENAI_API_KEY is not set. Cloud LLM features will be unavailable. "
        "Set it in your .env file or shell environment."
    )

# Neo4j credentials (also used by import_to_neo4j.py and app.py)
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

if not NEO4J_PASSWORD:
    logger.warning(
        "NEO4J_PASSWORD is not set. Database connections will likely fail. "
        "Set it in your .env file or via the NEO4J_PASSWORD environment variable."
    )

# ---------------------------------------------------------------------------
# Extraction Parameters
# ---------------------------------------------------------------------------
# Local LLM inference is memory-intensive; start with MAX_WORKERS=1
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "1"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2048"))
