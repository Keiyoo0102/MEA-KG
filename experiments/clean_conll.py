import os
import re
import logging
from tqdm import tqdm

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'experiments/')
# Base Data Directory
DATA_ROOT = '../data'

# Input: Raw CoNLL export from annotation tool (e.g., Doccano)
# PLEASE NOTE: You need to place your exported file here manually.
INPUT_CONLL_FILE = os.path.join(DATA_ROOT, 'annotations', 'fin.annotations.conll.txt')

# Output: Cleaned dataset ready for splitting
OUTPUT_DIR = os.path.join(DATA_ROOT, 'experiments', 'datasets')
OUTPUT_CLEAN_FILE = os.path.join(OUTPUT_DIR, 'MeaBIO.txt')

# Error markers to filter out
FAILURE_MARKERS = (
    "# FAILED_TO_TAG_SENTENCE:",
    "# ERROR_PROCESSING_FILE:",
    "# INPUT_FILE_EMPTY",
    "# ERROR_NLTK_RESOURCE_MISSING:"
)

# --- Main Execution ---

def clean_failed_sentences():
    """
    Reads the raw CoNLL file, removes sentence blocks marked as failures during 
    pre-processing or annotation, and saves the clean data.
    """
    logging.info(f"--- Starting CoNLL Cleaning ---")
    logging.info(f"Input File: {INPUT_CONLL_FILE}")

    # 1. Check Input
    if not os.path.exists(INPUT_CONLL_FILE):
        logging.error(f"Input file not found: {INPUT_CONLL_FILE}")
        logging.error("Please export your annotations from Doccano and place the file at the path above.")
        return

    # 2. Prepare Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_blocks = 0
    valid_blocks = 0
    failed_blocks = 0

    try:
        # 3. Read Content
        with open(INPUT_CONLL_FILE, 'r', encoding='utf-8') as f_in:
            content = f_in.read()

        # Split by empty lines (standard CoNLL sentence separator)
        # Regex handles \n\n, \r\n\r\n etc.
        sentence_blocks = re.split(r'\n\s*\n', content)
        total_blocks = len(sentence_blocks)
        
        logging.info(f"Read {total_blocks} raw sentence blocks.")

        # 4. Filter and Write
        with open(OUTPUT_CLEAN_FILE, 'w', encoding='utf-8') as f_out:
            for block in tqdm(sentence_blocks, desc="Cleaning Data"):
                cleaned_block = block.strip()

                # Skip empty blocks
                if not cleaned_block:
                    continue

                # Check for failure markers
                is_failed_block = False
                for marker in FAILURE_MARKERS:
                    if cleaned_block.startswith(marker):
                        is_failed_block = True
                        break

                if is_failed_block:
                    failed_blocks += 1
                    continue 

                # Valid block found
                valid_blocks += 1
                # Write block followed by two newlines (CoNLL standard separator)
                f_out.write(cleaned_block + "\n\n")

        logging.info("--- Cleaning Complete ---")
        logging.info(f"Total Blocks: {total_blocks}")
        logging.info(f"Valid Blocks Saved: {valid_blocks}")
        logging.info(f"Failed/Skipped Blocks: {failed_blocks}")
        logging.info(f"Cleaned dataset saved to: {OUTPUT_CLEAN_FILE}")

    except Exception as e:
        logging.error(f"An error occurred during cleaning: {e}")

if __name__ == '__main__':
    clean_failed_sentences()
