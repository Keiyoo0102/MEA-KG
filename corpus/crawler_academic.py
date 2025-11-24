import os
import re
import time
import glob
import logging
import pandas as pd
from tqdm import tqdm

# Import API clients
# NOTE: You need to install 'ads' and 'pyalex' first
# pip install ads pyalex
import ads
import pyalex
from pyalex import Works

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Credentials (Load from Environment Variables)
NASA_ADS_KEY = os.getenv("NASA_ADS_KEY", "YOUR_ADS_KEY_HERE")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "your_email@example.com") # For polite pool

# Configure Libraries
ads.config.token = NASA_ADS_KEY
pyalex.config.email = OPENALEX_EMAIL

# API Rate Limits
ADS_API_DELAY = 0.1
OPENALEX_API_DELAY_PER_PAGE = 0.5
OPENALEX_API_DELAY_PER_KEYWORD = 2.0

# --- Paths (Relative to 'corpus/') ---
# Input: Ontology concepts to generate search keywords
CONCEPTS_FILE = '../data/outputs/concepts_final_restructured_AI_DRAFT.csv'

# Outputs
OUTPUT_BASE_DIR = '../data/corpus_raw/academic'
OUTPUT_TEXT_DIR = os.path.join(OUTPUT_BASE_DIR, 'txt')
OUTPUT_METADATA_DIR = os.path.join(OUTPUT_BASE_DIR, 'metadata')
# Metadata filename prefix
OUTPUT_METADATA_BASENAME = os.path.join(OUTPUT_METADATA_DIR, 'academic_corpus_metadata')

# Metadata Columns
METADATA_HEADERS = ['file_id', 'doi', 'title', 'abstract', 'authors', 'year',
                    'keywords_ads', 'openalex_id', 'source_api', 'search_concept']

# Excel Row Limit Safety (Split CSVs if they get too big)
MAX_ROWS_PER_CSV = 1000000

# --- Search Keywords Configuration ---
# Macro Keywords (Broad concepts)
MACRO_KEYWORDS = [
    "Mars Earth Analog",
    "Mars Terrestrial Analogs",
    "Comparative Planetology Mars Earth",
]

# Search Query Modifiers
# ADS Syntax: abs:Earth AND (abs:Mars OR abs:Martian)
EARTH_MARS_TERMS_ADS = "abs:Earth AND (abs:Mars OR abs:Martian)"
# OpenAlex Syntax: earth "mars OR martian"
EARTH_MARS_TERMS_OPENALEX = 'earth "mars OR martian"'

# --- Helper Functions ---

def get_csv_segment_path(basename, index):
    """Generates filename for CSV segments (handling large datasets)."""
    if index == 0:
        return f"{basename}.csv"
    else:
        return f"{basename}_{index}.csv"

def count_rows_in_csv(filepath):
    """Efficiently counts rows in a CSV file."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            row_count = sum(1 for row in f) - 1
            return max(0, row_count)
    except FileNotFoundError:
        return 0
    except Exception as e:
        logging.error(f"Error counting rows in {filepath}: {e}")
        return 0

def load_existing_data(basename):
    """
    Loads existing CSV segments to support Resume Capability.
    Returns: processed_dois (set), processed_keywords (set), current paths/counters.
    """
    processed_dois = set()
    processed_keywords = set()
    current_csv_path = get_csv_segment_path(basename, 0)
    current_row_count = 0
    csv_segment_index = 0

    # Find all existing segments
    csv_files = sorted(
        glob.glob(f"{basename}*.csv"),
        key=lambda x: int(re.search(r'_(\d+)\.csv$', x).group(1)) if re.search(r'_(\d+)\.csv$', x) else 0
    )

    if not csv_files:
        logging.info(f"No existing metadata found. Creating new file: {current_csv_path}")
        os.makedirs(os.path.dirname(current_csv_path), exist_ok=True)
        pd.DataFrame(columns=METADATA_HEADERS).to_csv(current_csv_path, index=False, encoding='utf-8-sig')
    else:
        current_csv_path = csv_files[-1]
        # Extract index from filename
        match = re.search(r'_(\d+)\.csv$', current_csv_path)
        csv_segment_index = int(match.group(1)) if match else 0
        
        logging.info(f"Found {len(csv_files)} existing segments. Current: {current_csv_path}")
        logging.info("Loading processed IDs to resume work...")
        
        chunksize = 100000
        for file_path in tqdm(csv_files, desc="Loading history"):
            try:
                for chunk in pd.read_csv(file_path, usecols=['doi', 'search_concept'], chunksize=chunksize, low_memory=False):
                    # Add DOIs
                    processed_dois.update(chunk['doi'].astype(str).dropna())
                    # Add Keywords (from rows with data and placeholder rows)
                    processed_keywords.update(chunk['search_concept'].astype(str).dropna())
            except Exception as e:
                logging.error(f"Error reading chunk from {file_path}: {e}")

        logging.info(f"Resume state loaded: {len(processed_dois)} DOIs, {len(processed_keywords)} Keywords processed.")
        current_row_count = count_rows_in_csv(current_csv_path)

    return processed_dois, processed_keywords, current_csv_path, current_row_count, csv_segment_index

def reconstruct_abstract(inverted_abstract):
    """Reconstructs abstract text from OpenAlex inverted index format."""
    if not inverted_abstract: return None
    try:
        # Find the maximum index to determine list size
        max_index = max(max(indices) for indices in inverted_abstract.values() if indices)
        abstract_list = [""] * (max_index + 1)
        
        for word, indices in inverted_abstract.items():
            for index in indices:
                if 0 <= index < len(abstract_list):
                    abstract_list[index] = word
        return " ".join(filter(None, abstract_list))
    except Exception as e:
        # Suppress spammy errors, abstract might be malformed
        return None

def save_paper_data(paper_data, file_counter, current_csv_path, current_row_count, csv_segment_index, max_rows):
    """
    Saves metadata to CSV and abstract text to TXT file.
    Handles CSV segmentation if max_rows is reached.
    """
    try:
        # --- 1. Check CSV Split ---
        if current_row_count >= max_rows:
            logging.info(f"CSV limit reached ({current_row_count}). Creating new segment.")
            csv_segment_index += 1
            current_csv_path = get_csv_segment_path(OUTPUT_METADATA_BASENAME, csv_segment_index)
            pd.DataFrame(columns=METADATA_HEADERS).to_csv(current_csv_path, index=False, encoding='utf-8-sig')
            current_row_count = 0
            logging.info(f"Switched to new file: {current_csv_path}")

        # --- 2. Save TXT File ---
        file_id = f"acad_doc_{file_counter:06d}.txt"
        paper_data['file_id'] = file_id
        
        txt_content = f"{paper_data.get('title', 'No Title')}\n\n{paper_data.get('abstract', 'No Abstract')}"
        txt_file_path = os.path.join(OUTPUT_TEXT_DIR, file_id)
        
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)

        # --- 3. Save Metadata ---
        metadata_row = {h: paper_data.get(h) for h in METADATA_HEADERS}
        temp_meta_df = pd.DataFrame([metadata_row])
        temp_meta_df.to_csv(current_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        current_row_count += 1
        
        # Return updated state
        return file_counter + 1, current_csv_path, current_row_count, csv_segment_index

    except Exception as e:
        logging.error(f"Failed to save paper (DOI: {paper_data.get('doi')}): {e}")
        return file_counter, current_csv_path, current_row_count, csv_segment_index

def search_ads(keyword, is_macro_keyword, processed_dois_set):
    """Query NASA ADS API."""
    new_papers = []
    try:
        if is_macro_keyword:
            query = f'abs:"{keyword}" OR keyword:"{keyword}"'
        else:
            # Restrict specific concepts to Earth-Mars context
            query = f'(abs:"{keyword}" OR keyword:"{keyword}") AND {EARTH_MARS_TERMS_ADS}'

        search_results = ads.SearchQuery(
            q=query, 
            fl=['doi', 'title', 'abstract', 'author', 'year', 'keyword'],
            sort="citation_count desc"
        )

        for paper in search_results:
            # Clean DOI
            doi = paper.doi[0].lower() if paper.doi and paper.doi[0] else None
            
            if doi and doi not in processed_dois_set:
                paper_data = {
                    'doi': doi, 
                    'title': paper.title[0] if paper.title else None,
                    'abstract': paper.abstract if paper.abstract else None,
                    'authors': "; ".join(paper.author) if paper.author else None,
                    'year': int(paper.year) if paper.year else None,
                    'keywords_ads': "; ".join(paper.keyword) if paper.keyword else None,
                    'openalex_id': None, 
                    'source_api': 'ADS', 
                    'search_concept': keyword
                }
                if paper_data['abstract']:
                    new_papers.append(paper_data)

        logging.info(f"ADS found {len(new_papers)} new papers for '{keyword}'.")

    except ads.exceptions.APIResponseError as e:
        logging.error(f"ADS API Error ('{keyword}'): {e}")
        if "rate limit" in str(e).lower():
            logging.warning("ADS Rate Limit Hit! Sleeping for 1 hour...")
            time.sleep(3601)
    except Exception as e:
        logging.error(f"ADS Unknown Error ('{keyword}'): {e}")
        time.sleep(5)

    time.sleep(ADS_API_DELAY)
    return new_papers

def search_openalex(keyword, is_macro_keyword, processed_dois_set):
    """Query OpenAlex API."""
    new_papers = []
    page_count = 0
    try:
        if is_macro_keyword:
            search_term = keyword
        else:
            search_term = f'"{keyword}" {EARTH_MARS_TERMS_OPENALEX}'
            
        # Use cursor pagination for large results
        works_pager = Works().filter(
            abstract={'search': search_term}
        ).select([
            'id', 'doi', 'title', 'publication_year', 'authorships', 'abstract_inverted_index'
        ]).paginate(per_page=200, n_max=None, method="cursor")

        for page in works_pager:
            page_count += 1
            logging.info(f"OpenAlex '{keyword}' - Page {page_count}...")

            for work in page:
                raw_doi = work.get('doi')
                doi = raw_doi.replace("https://doi.org/", "").lower() if raw_doi else None

                if doi and doi not in processed_dois_set:
                    abstract = reconstruct_abstract(work.get('abstract_inverted_index'))
                    if abstract:
                        # Parse authors
                        authors = []
                        if work.get('authorships'):
                            authors = [a['author']['display_name'] for a in work['authorships'] 
                                      if a.get('author') and a.get('author').get('display_name')]
                        
                        paper_data = {
                            'doi': doi, 
                            'title': work.get('title'), 
                            'abstract': abstract,
                            'authors': "; ".join(authors) if authors else None,
                            'year': work.get('publication_year'), 
                            'keywords_ads': None,
                            'openalex_id': work.get('id'), 
                            'source_api': 'OpenAlex', 
                            'search_concept': keyword
                        }
                        new_papers.append(paper_data)

            # Respect rate limits
            time.sleep(OPENALEX_API_DELAY_PER_PAGE)
            # Limit depth for demo purposes (optional, remove break to fetch all)
            if page_count >= 5: break 

        logging.info(f"OpenAlex found {len(new_papers)} new papers for '{keyword}'.")

    except Exception as e:
        logging.error(f"OpenAlex Error ('{keyword}'): {e}")
        time.sleep(5)

    time.sleep(OPENALEX_API_DELAY_PER_KEYWORD)
    return new_papers

# --- Main Execution ---

def main():
    # 1. Setup Directories
    os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
    
    # 2. Check API Key
    if not NASA_ADS_KEY or "YOUR_ADS_KEY" in NASA_ADS_KEY:
        logging.error("Invalid ADS API Key. Please set 'NASA_ADS_KEY' in environment variables.")
        # We return but allow script to continue to try OpenAlex or just to show setup is needed
        return

    # 3. Load Keywords
    csv_keywords = []
    if os.path.exists(CONCEPTS_FILE):
        try:
            concepts_df = pd.read_csv(CONCEPTS_FILE)
            # Exclude root nodes (Scale names)
            filtered_df = concepts_df[~concepts_df['concept_name'].isin(['MacroScale', 'MesoScale', 'MicroScale'])]
            raw_keywords = filtered_df['concept_name'].unique()
            # Clean keywords
            csv_keywords = [re.sub(r'[^a-zA-Z0-9\s-]', '', k).strip() for k in raw_keywords if k]
            csv_keywords = list(filter(None, csv_keywords))
            logging.info(f"Loaded {len(csv_keywords)} specific concepts from ontology.")
        except Exception as e:
            logging.error(f"Error loading ontology file: {e}")
    else:
        logging.warning(f"Ontology file not found at {CONCEPTS_FILE}. Only using macro keywords.")

    # Combine and deduplicate
    all_keywords = list(set(MACRO_KEYWORDS + csv_keywords))
    logging.info(f"Total unique keywords to process: {len(all_keywords)}")

    # 4. Load State (Resume)
    processed_dois, processed_keywords, current_csv_path, current_row_count, csv_segment_index = load_existing_data(OUTPUT_METADATA_BASENAME)
    
    # Determine next file ID based on existing files
    existing_txts = glob.glob(os.path.join(OUTPUT_TEXT_DIR, 'acad_doc_*.txt'))
    if existing_txts:
        max_num = max([int(re.search(r'(\d+)', os.path.basename(f)).group(1)) for f in existing_txts])
        file_counter = max_num + 1
    else:
        file_counter = 1
    
    # 5. Process Keywords
    # Filter out already processed keywords
    keywords_to_process = [k for k in all_keywords if k not in processed_keywords]
    
    if not keywords_to_process:
        logging.info("All keywords have been processed! Exiting.")
        return

    logging.info(f"Starting processing for {len(keywords_to_process)} new keywords...")

    for keyword in tqdm(keywords_to_process, desc="Crawling Keywords"):
        is_macro = keyword in MACRO_KEYWORDS
        found_papers = []

        # --- Search ADS ---
        found_papers.extend(search_ads(keyword, is_macro, processed_dois))
        
        # --- Search OpenAlex ---
        found_papers.extend(search_openalex(keyword, is_macro, processed_dois))

        # --- Save Results ---
        unique_batch_papers = []
        batch_dois = set()

        for p in found_papers:
            doi = p['doi']
            if doi not in processed_dois and doi not in batch_dois:
                unique_batch_papers.append(p)
                batch_dois.add(doi)

        if unique_batch_papers:
            logging.info(f"Saving {len(unique_batch_papers)} unique papers for '{keyword}'...")
            for paper in unique_batch_papers:
                file_counter, current_csv_path, current_row_count, csv_segment_index = save_paper_data(
                    paper, file_counter, current_csv_path, current_row_count, csv_segment_index, MAX_ROWS_PER_CSV
                )
                processed_dois.add(paper['doi'])
        else:
            logging.info(f"No new unique papers found for '{keyword}'.")

        # --- Mark Keyword as Processed ---
        # Write a placeholder row to CSV to track progress even if no papers found
        try:
            placeholder = pd.DataFrame([{'search_concept': keyword}])
            placeholder.to_csv(current_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            current_row_count += 1
            # Check split again
            if current_row_count >= MAX_ROWS_PER_CSV:
                csv_segment_index += 1
                current_csv_path = get_csv_segment_path(OUTPUT_METADATA_BASENAME, csv_segment_index)
                pd.DataFrame(columns=METADATA_HEADERS).to_csv(current_csv_path, index=False, encoding='utf-8-sig')
                current_row_count = 0
        except Exception as e:
            logging.error(f"Error saving checkpoint for keyword '{keyword}': {e}")

    logging.info("--- Crawler Finished ---")

if __name__ == '__main__':
    main()
