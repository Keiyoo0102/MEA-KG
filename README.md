
# MEA-KG: Mars–Earth Analog Knowledge Graph 🪐

**An Automated Knowledge Graph Construction and Graph-CoT Reasoning Framework
for Comparative Planetology via Ontology-Guided Large Language Models.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Computers & Geosciences](https://img.shields.io/badge/Journal-Computers%20%26%20Geosciences-orange)]()

---

## 📖 Abstract

**MEA-KG** is a comprehensive open-source framework that bridges the knowledge
gap between Earth and Mars geology by automatically constructing and reasoning
over a comparative planetology knowledge graph.

The pipeline leverages Large Language Models (LLMs) guided by a domain-specific
ontology to extract structured knowledge from heterogeneous scientific texts
(peer-reviewed papers, NASA/ESA mission reports, and geology news).  The
extracted knowledge is stored in a dual-layer Neo4j graph combining an ontology
*Schema Layer* with an instance *Data Layer*.

The centrepiece of the reasoning system is the **MEA-Reasoner**, which
implements **Graph Chain-of-Thought (Graph-CoT)** — a three-stage white-box
inference mechanism that grounds LLM responses in retrieved graph evidence,
eliminating hallucination for planetary geology QA.


---

## 📂 Project Structure

```text
MEA-KG/
│
├── 📂 ontology/                   # Phase 1: Ontology Engineering
│   ├── 1_concept_extraction.py    # Extract raw concepts from PDFs / URLs
│   ├── 2_concept_alignment.py     # Semantic alignment using SBERT
│   ├── 3_review_duplicates.py     # AI-assisted de-duplication
│   ├── 4_structure_generation.py  # Generate Macro / Meso / Micro hierarchy
│   ├── 5_property_generation.py   # Generate object & data properties
│   └── 6_generate_owl.py          # Export final ontology to OWL format
│
├── 📂 corpus/                     # Phase 2a: Corpus Construction
│   ├── crawler_academic.py        # NASA ADS / OpenAlex paper crawler
│   ├── crawler_news.py            # Google News crawler (Selenium)
│   ├── crawler_web.py             # NASA / USGS agency website crawler
│   ├── preprocess.py              # Text cleaning & sentence segmentation
│   └── select_annotation_data.py  # Sample data for BIO annotation
│
├── 📂 experiments/                # Phase 2b: NER Model Evaluation
│   ├── clean_conll.py             # Clean annotated BIO data
│   ├── split_dataset.py           # Train / Dev / Test split
│   ├── eval_bert_crf.py           # Baseline: BERT-CRF
│   ├── eval_bert_lstm_crf.py      # Baseline: BERT-LSTM-CRF
│   ├── eval_bert_lstm.py          # Baseline: BERT-LSTM
│   ├── eval_roberta_crf.py        # Baseline: RoBERTa-CRF
│   ├── eval_lstm_crf.py           # Baseline: LSTM-CRF
│   ├── eval_llm_4omini.py         # GPT-4o-mini few-shot NER
│   ├── eval_llm_gpt-oss_20b.py    # GPT-OSS:20b fine-tuned NER
│   └── eval_mea_reasoner.py       # ← NEW: Graph-CoT evaluation script
│
├── 📂 pipeline/                   # Phase 2c: Full Extraction Pipeline
│   ├── mea_kg_builder/            # Core extraction package
│   │   ├── config.py              # Environment-based configuration
│   │   ├── llm_client.py          # Robust LLM client (tenacity retry)
│   │   ├── ontology_loader.py     # OWL parser & constraint manager
│   │   ├── extractor.py           # Dual-pipeline extraction engine
│   │   └── prompt_templates.py    # System prompt library
│   ├── main_extraction.py         # Parallel batch extraction entry point
│   ├── import_to_neo4j.py         # Import JSONL triplets into Neo4j
│   └── finetune_qlora.py          # ← NEW: QLoRA fine-tuning script
│
├── 📂 reasoner/                   # ← NEW: Graph-CoT Reasoning Package
│   ├── __init__.py
│   └── graph_cot.py               # Three-stage Graph-CoT implementation
│
├── 📂 application/                # Phase 3: Interactive Application
│   └── app.py                     # Streamlit dashboard (5 modules)
│
├── 📂 data/
│   ├── dummy_sample/              # ← NEW: Synthetic data for reproducibility
│   │   ├── sample_ontology.json   # Minimal synthetic ontology subset
│   │   ├── sample_corpus.txt      # Synthetic science text corpus
│   │   └── qa_eval_dataset.json   # 6-item QA benchmark (L1/L2/L3)
│   ├── ontology/                  # Place MEA_Ontology.owl here
│   └── corpus_preprocessed/       # Place cleaned .txt corpus files here
│
├── .env.example                   # ← NEW: Environment variable template
├── .gitignore
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned dependency versions
└── README.md
```

---

## 🛠️ Dependencies & Computational Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| Neo4j | ≥ 5.14 | Graph database |
| streamlit | ≥ 1.33 | Web dashboard |
| openai | ≥ 1.30 | LLM API client |
| transformers | ≥ 4.40 | NER model training |
| peft | ≥ 0.10 | LoRA / QLoRA adapters |
| sentence-transformers | ≥ 2.7 | Semantic alignment |
| langchain | ≥ 0.2 | KBQA chain orchestration |

### GPU Requirements for QLoRA Fine-Tuning (Section 2.4)

| Component | Requirement |
|-----------|-------------|
| GPU VRAM (minimum) | **16 GB** |
| GPU VRAM (recommended) | 24–40 GB |
| Tested hardware | NVIDIA A100 40 GB, RTX 3090 24 GB |
| CUDA version | ≥ 11.8 |
| bitsandbytes | ≥ 0.41 |

> **Note:** QLoRA 4-bit quantisation reduces the effective VRAM requirement
> for a 20B-parameter model from ~40 GB to approximately 12–16 GB.

The Graph-CoT reasoner and the Streamlit dashboard **do not require a GPU**
and can run on any machine with internet access to an OpenAI-compatible API.

---

## 🚀 Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Keiyoo0102/MEA-KG.git
cd MEA-KG
```

### Step 2 — Create a Virtual Environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For QLoRA fine-tuning (GPU required):

```bash
pip install bitsandbytes>=0.41.0 trl>=0.8.0 accelerate>=0.27.0
```

### Step 4 — Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env and fill in your credentials
# (Never commit .env to version control)
```

Required variables:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Step 5 — Start Neo4j

Download and start [Neo4j Desktop](https://neo4j.com/download/) or use Docker:

```bash
docker run \
  --name mea-kg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5.14
```

---

## 🔁 Reproducibility Guide

### Option A — Minimal Test with Dummy Dataset (No GPU, ~5 min)

This path uses the synthetic data in `data/dummy_sample/` to verify the
full pipeline without downloading any external data or running fine-tuning.

#### 1. Smoke-test the Graph-CoT reasoner (no Neo4j required)

```bash
# The reasoner falls back gracefully if Neo4j is unavailable
python reasoner/graph_cot.py \
  --query "What polygonal terrain patterns exist in Utopia Planitia?"
```

#### 2. Run the evaluation script on the dummy QA dataset

```bash
python experiments/eval_mea_reasoner.py \
  --dataset data/dummy_sample/qa_eval_dataset.json \
  --max_items 3
```

#### 3. Run the ablation experiment (w/o Analogy Rel)

```bash
python experiments/eval_mea_reasoner.py \
  --dataset data/dummy_sample/qa_eval_dataset.json \
  --mask_analogy \
  --max_items 3
```

#### 4. Test QLoRA fine-tuning code path (CPU-safe, no GPU needed)

```bash
python pipeline/finetune_qlora.py --dummy
```

Expected output: LoRA configuration printout and 2 formatted training
examples. No actual training occurs; this verifies the code logic only.

#### 5. Launch the Streamlit dashboard

```bash
cd application
streamlit run app.py
```

Navigate to `http://localhost:8501` and select the
**"🧠 MEA-Reasoner (Graph-CoT)"** module to interact with the reasoner.

---

### Option B — Full Pipeline Reproduction

#### Step 1 — Ontology Engineering

```bash
cd ontology
python 1_concept_extraction.py   # Extract concepts from PDFs
python 2_concept_alignment.py    # Align with SBERT
python 3_review_duplicates.py    # Deduplicate
python 4_structure_generation.py # Build hierarchy
python 5_property_generation.py  # Generate properties
python 6_generate_owl.py         # Export OWL
```

Output: `data/ontology/MEA_Ontology.owl`

#### Step 2 — Corpus Construction

```bash
cd corpus
python crawler_academic.py   # NASA ADS / OpenAlex
python crawler_news.py       # Google News
python crawler_web.py        # Agency websites
python preprocess.py         # Clean and segment
```

Output: `data/corpus_preprocessed/{academic,news,web}/*.txt`

#### Step 3 — QLoRA Fine-Tuning (GPU required)

```bash
# Prepare BIO annotation data first, then convert to instruction format
python pipeline/finetune_qlora.py \
  --data_path data/finetune/train.json \
  --model meta-llama/Llama-2-13b-hf \
  --epochs 3
```

Output: `data/experiments/lora_output/` (LoRA adapter weights)

Then create an Ollama model from the adapter:

```bash
# 1. Convert to GGUF format (requires llama.cpp)
python llama.cpp/convert_hf_to_gguf.py data/experiments/lora_output

# 2. Create and register the Ollama model
ollama create mea-kg-model -f pipeline/Modelfile
```

#### Step 4 — Knowledge Extraction

```bash
cd pipeline
python main_extraction.py
```

Output: `data/knowledge_graph_build/extraction_results.jsonl`

#### Step 5 — Graph Population

```bash
python pipeline/import_to_neo4j.py
```

#### Step 6 — Full Evaluation

```bash
# Full Graph-CoT evaluation
python experiments/eval_mea_reasoner.py \
  --dataset data/eval/qa_benchmark.json \
  --output_dir data/eval_results

# Ablation (w/o Analogy Rel)
python experiments/eval_mea_reasoner.py \
  --dataset data/eval/qa_benchmark.json \
  --mask_analogy \
  --output_dir data/eval_results

# Compare the two runs
python experiments/eval_mea_reasoner.py \
  --compare data/eval_results/eval_detail_full.json \
            data/eval_results/eval_detail_no_analogy.json
```

---

## 📚 User Guide — MEA-Reasoner Tutorial

### What is Graph-CoT?

Graph Chain-of-Thought (Graph-CoT) is the white-box reasoning mechanism that
distinguishes MEA-Reasoner from standard LLM QA systems.  Instead of relying
solely on LLM parametric knowledge (which may hallucinate), Graph-CoT:

1. **Extracts entities** from your query using an LLM.
2. **Retrieves a subgraph** from Neo4j via fuzzy matching on those entities.
3. **Synthesises a hypothesis** grounded strictly in the retrieved graph evidence.

### Input Format

Any natural-language question about planetary geology, e.g.:

- `"What polygonal terrain patterns exist in Utopia Planitia?"`
- `"How does the mineral composition of Gale Crater compare to Earth analogs?"`
- `"What processes formed the layered deposits in Valles Marineris?"`

### Output Structure

```json
{
  "query": "...",
  "extracted_entities": ["Utopia Planitia", "polygonal terrain", ...],
  "subgraph_triplets": [
    {"subject": "Utopia Planitia polygonal terrain",
     "relation": "ANALOG_OF",
     "object": "Siberian tundra polygon network",
     "is_analogy": true}
  ],
  "hypothesis": "**[Grounded Hypothesis]** ...\n**[Reasoning Chain]** ...",
  "mask_analogy": false
}
```

### Ablation Study (Reproducing Table 3 in the Manuscript)

To measure the contribution of Mars–Earth analogy relationships:

```python
from reasoner import GraphCoT

reasoner = GraphCoT()
query = "What polygonal terrain patterns exist in Utopia Planitia?"

# Full system (includes analogy edges)
result_full = reasoner.run(query, mask_analogy=False)

# Ablation: w/o Analogy Rel
result_ablation = reasoner.run(query, mask_analogy=True)

# Compare semantic coverage
print("Full Cov_sem:", result_full.semantic_coverage)      # higher
print("Ablation Cov_sem:", result_ablation.semantic_coverage)  # lower
```

### Dashboard Usage

| Module | Description |
|--------|-------------|
| 📊 System Overview | Global graph statistics and 2000-node visualisation |
| 🔍 Semantic Search | Fuzzy entity search with 1-hop neighbourhood graph |
| 🕸️ Exploratory Query | Browse entities by ontology type |
| ❓ Knowledge QA | LangChain-based KBQA with Cypher generation |
| 🧠 MEA-Reasoner | Interactive Graph-CoT with white-box stage display |

In the **MEA-Reasoner** tab, you can:
- Enter any planetary geology question.
- Toggle **"Ablation: w/o Analogy Rel"** to reproduce the ablation condition.
- Inspect each stage (entity list, subgraph visualisation, hypothesis).
- Download the full result as a JSON file.

---

## 📊 Reasoning Level Definitions

The QA benchmark categorises questions by reasoning difficulty:

| Level | Name | Description | Example |
|-------|------|-------------|---------|
| L1 | Direct Fact Retrieval | Single-hop graph lookup | "What minerals are in Gale Crater?" |
| L2 | Comparative Reasoning | Multi-hop + analogy edges | "How does Siberian tundra inform Utopia Planitia geology?" |
| L3 | Hypothesis Generation | Complex multi-entity synthesis | "Propose an ISRU strategy based on Utopia Planitia ice distribution" |

---

## 🔒 Security & Privacy

- **No hardcoded credentials**: All API keys and database passwords are loaded
  from environment variables only.  See `.env.example` for the required keys.
- **`.env` is in `.gitignore`**: Never commit your `.env` file.
- **Data provenance**: The full MEA corpus is derived from publicly available
  scientific literature (NASA ADS, OpenAlex) and agency reports.  No personally
  identifiable information is included.

---

## 📝 Citation

If you use MEA-KG in your research, please cite:

```bibtex
@article{meakg2025,
  title   = {MEA-KG: An Automated Knowledge Graph Construction and Graph-CoT
             Reasoning Framework for Comparative Planetology},
  author  = {[Authors]},
  journal = {Computers \& Geosciences},
  year    = {2025},
  doi     = {[DOI]}
}
```

---

## 🤝 Contributing

Contributions are welcome.  Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes with clear messages.
4. Open a Pull Request describing your changes.

---

## 🛡️ License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

*Disclaimer: This project was developed for academic research purposes.
Data sourced from NASA, ESA, and scientific publishers belongs to their
respective copyright holders.  The synthetic dummy dataset is entirely
fictitious and does not represent real scientific findings.*
