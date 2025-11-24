import os
import re
import sys
import html
import logging
import nltk
from tqdm import tqdm

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'corpus/')
# Input: Raw text from the crawlers (stored in data/corpus_raw/<source>/txt)
BASE_INPUT_DIR = '../data/corpus_raw'

# Mapping: Output Subdirectory Name -> Input Source Directory
CORPUS_DIRS_MAP = {
    'academic': os.path.join(BASE_INPUT_DIR, 'academic', 'txt'),
    'news': os.path.join(BASE_INPUT_DIR, 'news', 'txt'),
    'web': os.path.join(BASE_INPUT_DIR, 'web', 'txt')
}

# Output: Preprocessed sentences
OUTPUT_BASE_DIR = '../data/corpus_preprocessed'

# Parameters
MIN_SENTENCE_LENGTH = 5

# --- Helper Functions ---

def clean_text(raw_text):
    """Performs basic text cleaning: unescape HTML, remove tags, normalize whitespace."""
    if not raw_text: return ""
    text = html.unescape(raw_text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def segment_sentences(text):
    """Segments text into sentences using NLTK."""
    try:
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences]
        # Filter short sentences (often noise)
        sentences = [s for s in sentences if len(s.split()) >= MIN_SENTENCE_LENGTH]
        return sentences
    except LookupError as e:
        # Re-raise to be handled by main resource checker
        raise e
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return []

def check_nltk_resources():
    """Ensures necessary NLTK data is downloaded."""
    required_resources = ['punkt', 'punkt_tab']
    
    for resource in required_resources:
        try:
            # Try to find it first
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'punkt_tab':
                # punkt_tab structure is slightly different, easier to just try download if unsure
                # or assume it's handled by punkt check usually. 
                # We'll try a passive download to be safe.
                nltk.download(resource, quiet=True)
            
            logging.info(f"NLTK resource '{resource}' is ready.")
        except LookupError:
            logging.info(f"NLTK resource '{resource}' not found. Downloading...")
            try:
                nltk.download(resource, quiet=True)
                logging.info(f"Downloaded '{resource}'.")
            except Exception as e:
                logging.error(f"Failed to download NLTK resource '{resource}': {e}")
                return False
    return True

# --- Main Execution ---

def main():
    # 1. Setup
    if not check_nltk_resources():
        sys.exit("Critical Error: NLTK resources missing.")

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    logging.info(f"Output directory ready: {OUTPUT_BASE_DIR}")

    total_files_processed = 0
    total_sentences_written = 0

    # 2. Process each source directory
    for output_subdir_name, source_dir in CORPUS_DIRS_MAP.items():
        logging.info(f"--- Processing Source: {output_subdir_name} ---")
        
        # Check input dir exists
        if not os.path.exists(source_dir):
            logging.warning(f"Source directory not found: {source_dir}. Skipping.")
            continue

        # Create output subdirectory (e.g., data/corpus_preprocessed/academic)
        output_subdir_path = os.path.join(OUTPUT_BASE_DIR, output_subdir_name)
        os.makedirs(output_subdir_path, exist_ok=True)

        # Get files
        try:
            txt_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".txt")]
        except Exception as e:
            logging.error(f"Error reading directory {source_dir}: {e}")
            continue

        if not txt_files:
            logging.warning(f"No .txt files found in {source_dir}.")
            continue

        logging.info(f"Found {len(txt_files)} files. Starting preprocessing...")

        # 3. File Loop
        for filename in tqdm(txt_files, desc=f"Cleaning {output_subdir_name}"):
            input_filepath = os.path.join(source_dir, filename)
            output_filepath = os.path.join(output_subdir_path, filename)
            
            sentences = []

            try:
                # Read
                with open(input_filepath, 'r', encoding='utf-8') as f_in:
                    raw_text = f_in.read()

                # Process
                cleaned_text = clean_text(raw_text)
                sentences = segment_sentences(cleaned_text)

                # Write
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    if sentences:
                        for sentence in sentences:
                            f_out.write(sentence + '\n')
                        total_sentences_written += len(sentences)
                    else:
                        # Write a marker if no valid sentences found
                        f_out.write("# PREPROCESSING_SKIPPED: NO_VALID_SENTENCES\n")

            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

            total_files_processed += 1

    # 4. Summary
    logging.info("--- Preprocessing Complete ---")
    logging.info(f"Total Files Processed: {total_files_processed}")
    logging.info(f"Total Sentences Written: {total_sentences_written}")
    logging.info(f"Results saved to: {OUTPUT_BASE_DIR}")

if __name__ == '__main__':
    main()
