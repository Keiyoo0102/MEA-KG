import os
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to the script location in 'ontology/')
# Input comes from step 1 output
INPUT_FILE_PATH = '../data/outputs/source_concepts_final.csv'
OUTPUT_FILE_PATH = '../data/outputs/concepts_to_review.csv'

# Model Settings
# Using a high-performance model optimized for semantic similarity
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Similarity Threshold (0.0 to 1.0)
# Concept pairs with cosine similarity higher than this will be flagged for review
SIMILARITY_THRESHOLD = 0.90

# --- Main Execution Flow ---

def main():
    """
    Main function for Concept Alignment.
    
    Steps:
    1. Load raw concepts extracted in Step 1.
    2. Generate vector embeddings for descriptions using SBERT.
    3. Mine for semantic duplicates or highly similar concepts.
    4. Export pairs for manual or AI-assisted review.
    """
    
    # 1. Check Input
    if not os.path.exists(INPUT_FILE_PATH):
        logging.error(f"Input file not found: {INPUT_FILE_PATH}")
        logging.error("Please run 'ontology/1_concept_extraction.py' first to generate data.")
        return

    # 2. Load Data
    logging.info(f"Loading concepts from {INPUT_FILE_PATH}...")
    try:
        df = pd.read_csv(INPUT_FILE_PATH)
    except Exception as e:
        logging.error(f"Failed to read input CSV: {e}")
        return

    # Data Cleaning: Remove empty descriptions and ensure string type
    df.dropna(subset=['description'], inplace=True)
    df['description'] = df['description'].astype(str)
    
    # Generate temporary IDs for tracking if not present
    if 'concept_id' not in df.columns:
        df['concept_id'] = df.index

    logging.info(f"Successfully loaded {len(df)} valid concepts.")

    # 3. Load SBERT Model
    logging.info(f"Loading pre-trained model '{MODEL_NAME}'... (This may take a moment)")
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model: {e}")
        logging.error("Try running: pip install sentence-transformers")
        return

    # 4. Semantic Paraphrase Mining
    logging.info("Preparing descriptions for embedding...")
    descriptions = df['description'].tolist()

    logging.info(f"Mining for pairs with semantic similarity > {SIMILARITY_THRESHOLD}...")
    
    # util.paraphrase_mining is highly optimized for finding duplicates in large lists
    # It returns a list of tuples: (score, i, j)
    similar_pairs = util.paraphrase_mining(
        model, 
        descriptions, 
        top_k=5, 
        corpus_chunk_size=5000,
        score_function=util.cos_sim
    )

    logging.info(f"Mining processing complete. Filtering results...")

    # 5. Format Results
    results = []
    for score, i, j in similar_pairs:
        if score >= SIMILARITY_THRESHOLD:
            # Indices i and j correspond to the 'descriptions' list and the DataFrame rows
            concept1 = df.iloc[i]
            concept2 = df.iloc[j]

            # Skip if names are identical (exact duplicates are trivial)
            if concept1['concept_name'] == concept2['concept_name']:
                continue

            results.append({
                'score': score,
                'concept_id_1': concept1['concept_id'],
                'concept_name_1': concept1['concept_name'],
                'concept_id_2': concept2['concept_id'],
                'concept_name_2': concept2['concept_name'],
                'description_1': concept1['description'],
                'description_2': concept2['description'],
            })

    if not results:
        logging.warning(f"No similar pairs found above threshold {SIMILARITY_THRESHOLD}.")
        logging.warning("You might want to lower the SIMILARITY_THRESHOLD in the script.")
        return

    # Sort by score descending (most similar first)
    review_df = pd.DataFrame(results)
    review_df = review_df.sort_values(by='score', ascending=False)

    # 6. Save Output
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    review_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
    
    logging.info(f"--- Alignment Flow Complete ---")
    logging.info(f"Found {len(review_df)} pairs requiring review.")
    logging.info(f"Results saved to: {OUTPUT_FILE_PATH}")

if __name__ == '__main__':
    main()
