import os
import random
import shutil
import logging
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'corpus/')
# Base Data Directory
DATA_ROOT = '../data'

# Input: Preprocessed Text Files
PREPROCESSED_DIR = os.path.join(DATA_ROOT, 'corpus_preprocessed')
SOURCE_DIRS = {
    'academic': os.path.join(PREPROCESSED_DIR, 'academic'),
    'news': os.path.join(PREPROCESSED_DIR, 'news'),
    'web': os.path.join(PREPROCESSED_DIR, 'web')
}

# Output: Directory for manual annotation (e.g., for Doccano)
ANNOTATION_SET_DIR = os.path.join(DATA_ROOT, 'corpus_for_annotation')
# Manifest file to track where files came from
MANIFEST_FILE = os.path.join(ANNOTATION_SET_DIR, 'annotation_manifest.csv')

# Sampling Parameters
TOTAL_ANNOTATION_FILES = 300  # Total number of files to select
# Stratified sampling proportions (must sum to 1.0)
PROPORTIONS = {
    'academic': 0.90, # 270 files
    'news': 0.09,     # 27 files
    'web': 0.01       # 3 files
}
RANDOM_SEED = 42

# --- Main Execution ---

def main():
    """
    Performs stratified random sampling of text files for manual annotation.
    Moves selected files to a separate directory and generates a manifest.
    """
    
    # 1. Validation
    if abs(sum(PROPORTIONS.values()) - 1.0) > 0.01:
        logging.error(f"Configuration Error: Proportions sum to {sum(PROPORTIONS.values())}, expected 1.0")
        return

    # 2. Prepare Output Directory
    if os.path.exists(ANNOTATION_SET_DIR):
        logging.warning(f"Output directory already exists: {ANNOTATION_SET_DIR}")
        user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
        
        if user_input == 'y':
            try:
                shutil.rmtree(ANNOTATION_SET_DIR)
                os.makedirs(ANNOTATION_SET_DIR)
                logging.info("Directory cleared and recreated.")
            except Exception as e:
                logging.error(f"Failed to clear directory: {e}")
                return
        else:
            logging.info("Operation cancelled by user.")
            return
    else:
        os.makedirs(ANNOTATION_SET_DIR)
        logging.info(f"Created output directory: {ANNOTATION_SET_DIR}")

    # 3. Set Seed
    random.seed(RANDOM_SEED)

    # 4. Scan Available Files
    source_files = {}
    total_available = 0
    
    logging.info("Scanning source directories...")
    for source_name, dir_path in SOURCE_DIRS.items():
        if not os.path.exists(dir_path):
            logging.warning(f"Source directory missing: {dir_path}")
            source_files[source_name] = []
            continue
            
        try:
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(".txt")]
            source_files[source_name] = files
            logging.info(f"[{source_name}] Found {len(files)} files.")
            total_available += len(files)
        except Exception as e:
            logging.error(f"Error scanning {dir_path}: {e}")
            source_files[source_name] = []

    if total_available == 0:
        logging.error("No files found to sample from.")
        return

    # 5. Perform Sampling
    selected_manifest = []
    total_sampled = 0
    
    logging.info(f"--- Starting Sampling (Target: {TOTAL_ANNOTATION_FILES}) ---")
    
    for source_name, files in source_files.items():
        if not files: continue

        # Calculate target count for this source
        target_count = int(round(TOTAL_ANNOTATION_FILES * PROPORTIONS.get(source_name, 0)))
        # Cap at available files
        actual_count = min(target_count, len(files))
        
        logging.info(f"Sampling {source_name}: Target {target_count}, Actual {actual_count}")
        
        # Random Sample
        sampled_files = random.sample(files, actual_count)
        
        # Copy Files
        for filename in tqdm(sampled_files, desc=f"Copying {source_name}"):
            src_path = os.path.join(SOURCE_DIRS[source_name], filename)
            dst_path = os.path.join(ANNOTATION_SET_DIR, filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                selected_manifest.append({
                    'filename': filename,
                    'original_source': source_name
                })
            except Exception as e:
                logging.error(f"Failed to copy {filename}: {e}")
                
        total_sampled += actual_count

    # 6. Save Manifest
    if selected_manifest:
        manifest_df = pd.DataFrame(selected_manifest)
        manifest_df.to_csv(MANIFEST_FILE, index=False, encoding='utf-8-sig')
        logging.info(f"Manifest saved to: {MANIFEST_FILE}")
    
    logging.info("--- Selection Complete ---")
    logging.info(f"Total Files Sampled: {total_sampled}")
    logging.info(f"Ready for annotation in: {ANNOTATION_SET_DIR}")

if __name__ == '__main__':
    main()
