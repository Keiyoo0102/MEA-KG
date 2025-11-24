
# MEA-KG: Mars-Earth Analog Knowledge Graph 🪐

**An Automated Knowledge Graph Construction Framework for Comparative Planetology via Ontology-Guided Large Language Models.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📖 Overview

**MEA-KG** is a comprehensive project designed to bridge the knowledge gap between Earth and Mars geology. By leveraging **Large Language Models (LLMs)** guided by a rigorous ontology, this framework automates the extraction, structuring, and visualization of comparative planetology knowledge from unstructured scientific texts, news, and agency reports.

### Key Features
- **Ontology Engineering**: Semi-automated ontology construction using LLMs and expert review, resulting in a hierarchical structure of concepts like *GeologicFeature*, *Process*, and *MaterialComposition*.
- **Hybrid Extraction Pipeline**: A dual-stage "Extract-Correct" pipeline that combines local models (Ollama/SBERT) with cloud LLMs (GPT-4o) for high precision and ontology compliance.
- **Dual-Layer Architecture**: Seamless integration of the **Schema Layer** (Ontology) and **Data Layer** (Instances) in Neo4j.
- **Interactive Application**: A Streamlit-based dashboard for global graph visualization, semantic search, and Knowledge-Based QA (KBQA).

---

## 📂 Project Structure

The repository is organized into modular components for clarity and maintainability:

```text
MEA-KG/
├── 📂 ontology/                 # Phase 1: Ontology Engineering
│   ├── 1_concept_extraction.py  # Extract raw concepts from PDFs/URLs
│   ├── 2_concept_alignment.py   # Semantic alignment using SBERT
│   ├── 3_review_duplicates.py   # AI-assisted de-duplication
│   ├── 4_structure_generation.py# Generate hierarchical structure (Macro/Meso/Micro)
│   ├── 5_property_generation.py # Generate object/data properties
│   └── 6_generate_owl.py        # Export final Ontology to OWL format
│
├── 📂 corpus/                   # Phase 2a: Corpus Construction
│   ├── crawler_academic.py      # Fetch papers from NASA ADS / OpenAlex
│   ├── crawler_news.py          # Fetch news from Google News via Selenium
│   ├── crawler_web.py           # Crawl official agency websites (NASA, USGS)
│   ├── preprocess.py            # Text cleaning and sentence segmentation
│   └── select_annotation_data.py# Sample data for BIO annotation
│
├── 📂 experiments/              # Phase 2b: NER Model Evaluation
│   ├── clean_conll.py           # Clean annotated BIO data
│   ├── split_dataset.py         # Split into Train/Dev/Test
│   ├── eval_bert_crf.py         # Baseline: BERT-CRF implementation
│   ├── eval_llm_rag.py          # Ours: LLM with RAG-based Few-Shot learning
│   └── ... (other model evaluations like RoBERTa, LSTM)
│
├── 📂 pipeline/                 # Phase 2c: Full Extraction Pipeline
│   ├── mea_kg_builder/          # Core Logic Package
│   │   ├── config.py            # Global configuration & Model settings
│   │   ├── llm_client.py        # Robust LLM client (instructor/tenacity)
│   │   ├── ontology_loader.py   # OWL Parser & Constraint Manager
│   │   ├── extractor.py         # Dual-Pipeline Extraction Engine
│   │   └── prompt_templates.py  # System prompts management
│   ├── main_extraction.py       # Main script for parallel batch processing
│   └── import_to_neo4j.py       # Import JSONL results into Neo4j
│
├── 📂 application/              # Phase 3: Application
│   └── app.py                   # Streamlit Dashboard (Visualization & KBQA)
│
├── 📂 data/                     # Data Placeholders
│   ├── ontology/                # Place MEA_Ontology.owl here
│   └── corpus_preprocessed/     # Place cleaned .txt files here
│
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
````

-----

## 🚀 Getting Started

### 1\. Prerequisites

  * **Python 3.9+**
  * **Neo4j Database** (Community or Enterprise, v5.x recommended).
  * **API Keys**: OpenAI API key (or compatible service like gptsapi.net) is required for the LLM extractor.
  * **Ollama** (Optional): If running local models like `gpt-oss:20b`.

### 2\. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/Keiyoo0102/MEA-KG.git](https://github.com/Keiyoo0102/MEA-KG.git)
cd MEA-KG
pip install -r requirements.txt
```

### 3\. Configuration

**⚠️ Important:** Do not commit your API keys.
Navigate to `pipeline/mea_kg_builder/config.py` and configure your settings:

```python
# Example configuration in config.py
ONTOLOGY_OWL_PATH = "../data/ontology/MEA_Ontology.owl"
CORPUS_DIR = "../data/corpus_preprocessed"
# Set API keys using environment variables is recommended
```

-----

## 🛠️ Construction Pipeline

### Step 1: Ontology Engineering

Run the scripts in `ontology/` sequentially to build the schema from scratch.

```bash
cd ontology
# 1. Extract concepts
python 1_concept_extraction.py
# ... (run intermediate steps) ...
# 6. Generate final OWL
python 6_generate_owl.py
```

*Output:* `MEA_Ontology.owl` stored in `data/ontology/`.

### Step 2: Corpus Preparation

Collect and preprocess data from multiple sources.

```bash
cd corpus
# Example: Crawl academic papers
python crawler_academic.py
# Preprocess text for extraction
python preprocess.py
```

### Step 3: Full Knowledge Extraction

Run the main parallel extraction engine. This uses the `mea_kg_builder` package to process text files.

```bash
cd pipeline
python main_extraction.py
```

*Output:* `extraction_results.jsonl` containing structured triplets.

### Step 4: Graph Population

Import the extracted triplets into Neo4j. Ensure your Neo4j instance is running.

```bash
python import_to_neo4j.py
```

-----

## 📊 Application Demo

Launch the interactive dashboard to explore the graph and ask questions.

```bash
cd application
streamlit run app.py
```

**Dashboard Features:**

1.  **System Overview**: Visualize topological statistics and the global graph structure (Red Schema nodes vs. Blue Data nodes).
2.  **Semantic Search**: Perform fuzzy searches for entities (e.g., "Gale Crater") and view their immediate neighbors.
3.  **Knowledge QA**: A chatbot interface that translates natural language questions into Cypher queries to answer questions like *"What hydrological processes occurred in Jezero Crater?"*.

-----

## 🛡️ License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome\! Please open an issue or submit a pull request.



*Disclaimer: This project was developed for academic research purposes. Data sources (NASA, ESA, etc.) belong to their respective owners.*

