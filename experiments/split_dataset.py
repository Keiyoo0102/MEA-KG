import os
import re
import random
import logging
from math import floor

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Paths (Relative to 'experiments/')
# Base Data Directory
DATA_ROOT = '../data'
DATASET_DIR = os.path.join(DATA_ROOT, 'experiments', 'datasets')

# Input: Cleaned CoNLL file from previous step
INPUT_CONLL_FILE = os.path.join(DATASET_DIR, 'MeaBIO.txt')

# Outputs: Split datasets
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.txt')
DEV_FILE = os.path.join(DATASET_DIR, 'dev.txt')   # Validation set
TEST_FILE = os.path.join(DATASET_DIR, 'test.txt') # Test set

# Split Ratios (Must sum to 1.0)
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

# --- Helper Functions ---

def write_blocks_to_file(filepath, blocks):
    """Writes a list of sentence blocks to a file, separated by newlines."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Join blocks with double newline (standard CoNLL separator)
            f.write("\n\n".join(blocks))
            # Ensure file ends with a newline if content exists
            if blocks:
                f.write("\n")
        logging.info(f"Saved {len(blocks)} sentences to: {filepath}")
    except Exception as e:
        logging.error(f"Error writing to {filepath}: {e}")

# --- Main Execution ---

def split_dataset():
    """
    Reads the cleaned CoNLL file, shuffles sentence blocks, 
    and splits them into Train, Dev, and Test sets.
    """
    logging.info("--- Starting Dataset Split ---")
    logging.info(f"Input File: {INPUT_CONLL_FILE}")

    # 1. Check Input
    if not os.path.exists(INPUT_CONLL_FILE):
        logging.error(f"Input file not found: {INPUT_CONLL_FILE}")
        logging.error("Please run 'experiments/clean_conll.py' first.")
        return

    # 2. Read and Parse
    try:
        with open(INPUT_CONLL_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by empty lines
        sentence_blocks = re.split(r'\n\s*\n', content)
        # Filter empty blocks
        sentence_blocks = [block.strip() for block in sentence_blocks if block.strip()]

        if not sentence_blocks:
            logging.error("No valid sentence blocks found in input file.")
            return

        logging.info(f"Total valid sentences found: {len(sentence_blocks)}")

    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    # 3. Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(sentence_blocks)
    logging.info(f"Data shuffled with seed {RANDOM_SEED}.")

    # 4. Calculate Splits
    total_count = len(sentence_blocks)
    train_count = floor(total_count * TRAIN_RATIO)
    dev_count = floor(total_count * DEV_RATIO)
    # Assign remaining to test to avoid rounding errors
    
    split_1 = train_count
    split_2 = train_count + dev_count

    # 5. Perform Split
    train_blocks = sentence_blocks[:split_1]
    dev_blocks = sentence_blocks[split_1:split_2]
    test_blocks = sentence_blocks[split_2:]

    logging.info("--- Split Statistics ---")
    logging.info(f"Train: {len(train_blocks)} ({TRAIN_RATIO*100}%)")
    logging.info(f"Dev:   {len(dev_blocks)} ({DEV_RATIO*100}%)")
    logging.info(f"Test:  {len(test_blocks)} ({TEST_RATIO*100}%)")

    # 6. Save Files
    write_blocks_to_file(TRAIN_FILE, train_blocks)
    write_blocks_to_file(DEV_FILE, dev_blocks)
    write_blocks_to_file(TEST_FILE, test_blocks)

    logging.info("--- Dataset Splitting Complete ---")

if __name__ == '__main__':
    split_dataset()
