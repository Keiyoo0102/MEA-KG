import os
import re
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm

# Third-party libraries
from gnews import GNews
import requests

# Selenium & Parsing setup
try:
    import undetected_chromedriver as uc
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    import newspaper
except ImportError as e:
    logging.error(f"Import Error: {e}")
    logging.error("Please ensure you have installed: 'undetected-chromedriver' and 'newspaper3k'")
    sys.exit(1)

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'corpus/')
# Output Directories
BASE_OUTPUT_DIR = '../data/corpus_raw/news'
OUTPUT_TEXT_DIR = os.path.join(BASE_OUTPUT_DIR, 'txt')
OUTPUT_METADATA_DIR = os.path.join(BASE_OUTPUT_DIR, 'metadata')
OUTPUT_METADATA_FILE = os.path.join(OUTPUT_METADATA_DIR, 'news_corpus_metadata.csv')

# Input: Ontology concepts for keyword generation
# Note: Using the file generated in Ontology Step 4
CONCEPTS_FILE = '../data/outputs/concepts_final_restructured_AI_DRAFT.csv'

# CSV Headers
METADATA_HEADERS = ['file_id', 'source_url', 'title', 'publisher', 'published_date', 'search_concept', 'text_saved']

# Search Settings
MACRO_KEYWORDS = [
    "Mars Earth Analog",
    "Mars Terrestrial Analogs",
    "Comparative Planetology Mars Earth",
]

# Crawler Settings
API_DELAY = 0.5
COUNTRY = 'US'
LANGUAGE = 'en'
MAX_RESULTS_PER_KW = 50
SELENIUM_WAIT_TIME = 0.5

# Browser Headers
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}

# Anti-Bot Detection Keywords
BOT_KEYWORDS = [
    "Verifying you are human", "CAPTCHA", "Cloudflare",
    "I am not a robot", "Please enable JavaScript", "Access Denied"
]

# --- Helper Functions ---

def load_existing_data():
    """Loads existing metadata to support Resume Capability."""
    visited_urls = set()
    processed_keywords = set()
    next_file_counter = 1
    
    # Ensure directory exists
    os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)

    file_exists = os.path.exists(OUTPUT_METADATA_FILE)

    if file_exists:
        try:
            logging.info(f"Loading existing metadata: {OUTPUT_METADATA_FILE}...")
            # Read existing data
            existing_df = pd.read_csv(OUTPUT_METADATA_FILE, usecols=METADATA_HEADERS[:-1]) 
            
            if not existing_df.empty:
                visited_urls = set(existing_df['source_url'].dropna())
                processed_keywords = set(existing_df['search_concept'].dropna())

                # Determine next file ID
                # Assumes file_id format 'news_doc_00001'
                numeric_file_ids = pd.to_numeric(
                    existing_df['file_id'].str.extract(r'(\d+)', expand=False),
                    errors='coerce'
                )
                if numeric_file_ids.notna().any():
                    next_file_counter = int(numeric_file_ids.max()) + 1

                logging.info(f"Loaded {len(visited_urls)} URLs and {len(processed_keywords)} keywords.")
                logging.info(f"Next File ID: {next_file_counter}")
        except pd.errors.EmptyDataError:
            logging.warning("Metadata file is empty.")
            file_exists = False
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            file_exists = False

    if not file_exists:
        # Create new file with headers
        pd.DataFrame(columns=METADATA_HEADERS).to_csv(OUTPUT_METADATA_FILE, index=False, encoding='utf-8-sig')
        logging.info("Created new metadata file.")

    return visited_urls, next_file_counter, processed_keywords

def sanitize_filename(name):
    """Sanitizes strings for use as filenames."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

def search_and_save_keyword(keyword, is_macro_keyword, google_news, visited_urls_set, file_counter_int, driver):
    """
    Searches for a keyword, crawls results using Selenium, and saves text/metadata.
    """
    current_file_counter = file_counter_int

    # Construct Search Query
    if is_macro_keyword:
        search_query = keyword
        log_keyword = keyword
    else:
        # Refine search for specific concepts to ensure context
        search_query = f'"{keyword}" Earth Mars Martian'
        log_keyword = keyword

    logging.info(f"--- Searching: '{search_query}' ---")

    try:
        news_items = google_news.get_news(search_query)

        if not news_items:
            logging.warning(f"No news found for '{search_query}'.")
            # Log placeholder to mark keyword as processed
            placeholder_metadata = {
                'file_id': 'N/A', 'source_url': f'search_term_{log_keyword}', 'title': 'No news found',
                'publisher': 'N/A', 'published_date': 'N/A', 'search_concept': log_keyword, 'text_saved': False
            }
            pd.DataFrame([placeholder_metadata]).to_csv(OUTPUT_METADATA_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
            return current_file_counter

        for item in news_items:
            google_url = item['url']

            # Skip visited
            if google_url in visited_urls_set:
                continue

            final_url = google_url
            html_content = None
            is_blocked = False

            try:
                # Selenium Navigation
                logging.info(f"Visiting: {google_url}")
                driver.get(google_url)
                time.sleep(SELENIUM_WAIT_TIME)
                
                final_url = driver.current_url
                html_content = driver.page_source
                
                # Simple Bot Check
                if any(bot_kw.lower() in html_content.lower() for bot_kw in BOT_KEYWORDS):
                    is_blocked = True
                    logging.warning(f"Potential Bot Block at: {final_url}")

            except Exception as e:
                logging.error(f"Selenium Error ({google_url}): {e}")
                # Continue to try next item

            # Double check final URL
            if final_url in visited_urls_set:
                visited_urls_set.add(google_url)
                continue

            # Prepare Metadata
            file_base_name = f"news_doc_{current_file_counter:05d}"
            file_name_txt = file_base_name + ".txt"
            text_saved_successfully = False

            new_metadata = {
                'file_id': file_base_name,
                'source_url': final_url,
                'title': item.get('title', 'N/A'),
                'publisher': item.get('publisher', {}).get('title', 'N/A'),
                'published_date': item.get('published date', 'N/A'),
                'search_concept': log_keyword,
                'text_saved': text_saved_successfully
            }

            # Extract Text using Newspaper3k
            if "youtube.com" in final_url or "youtu.be" in final_url:
                logging.warning("Skipping YouTube link.")
            elif is_blocked:
                logging.warning("Skipping blocked page.")
            elif html_content and len(html_content) > 150:
                try:
                    article = newspaper.Article(final_url)
                    article.set_html(html_content)
                    article.parse()

                    if article.text and len(article.text) >= 150:
                        clean_text = (article.title or "No Title") + "\n\n" + article.text
                        file_path = os.path.join(OUTPUT_TEXT_DIR, file_name_txt)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(clean_text)
                        
                        text_saved_successfully = True
                        logging.info(f"Saved: {file_name_txt}")
                    else:
                        logging.warning(f"Content too short or empty: {final_url}")
                except Exception as e:
                    logging.error(f"Newspaper Parsing Error: {e}")
            else:
                logging.warning("HTML content invalid or empty.")

            # Save Metadata (Append mode)
            new_metadata['text_saved'] = text_saved_successfully
            pd.DataFrame([new_metadata]).to_csv(OUTPUT_METADATA_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

            # Update state
            visited_urls_set.add(google_url)
            visited_urls_set.add(final_url)
            current_file_counter += 1

        time.sleep(API_DELAY)

    except Exception as e:
        logging.error(f"GNews Search Error ('{search_query}'): {e}")
        time.sleep(API_DELAY * 5)

    return current_file_counter

# --- Main Execution ---

def main():
    # 1. Setup Directories
    os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
    logging.info(f"Output directories ready.")

    # 2. Load State
    visited_urls, file_counter, processed_keywords = load_existing_data()

    # 3. Load Keywords from Ontology
    csv_keywords = []
    if os.path.exists(CONCEPTS_FILE):
        try:
            concepts_df = pd.read_csv(CONCEPTS_FILE)
            # Filter out Scale concepts
            valid_concepts = concepts_df[~concepts_df['concept_name'].isin(['MacroScale', 'MesoScale', 'MicroScale'])]
            raw_keywords = valid_concepts['concept_name'].unique()
            
            # Clean
            csv_keywords = [re.sub(r'[^a-zA-Z0-9\s-]', '', k).strip() for k in raw_keywords if k]
            csv_keywords = list(filter(None, csv_keywords))
            logging.info(f"Loaded {len(csv_keywords)} concept keywords.")
        except Exception as e:
            logging.error(f"Error reading ontology file: {e}")
    else:
        logging.warning(f"Ontology file not found at {CONCEPTS_FILE}. Proceeding with macro keywords only.")

    # 4. Initialize GNews
    try:
        google_news = GNews(language=LANGUAGE, country=COUNTRY, max_results=MAX_RESULTS_PER_KW)
    except Exception as e:
        logging.error(f"Failed to initialize GNews: {e}")
        return

    # 5. Initialize Selenium (Undetected Chromedriver)
    driver = None
    try:
        logging.info("Initializing Selenium (undetected_chromedriver)...")
        options = ChromeOptions()
        # Note: Headless mode is often detected, using normal mode with subprocess
        # options.add_argument('--headless') 
        options.add_argument('--disable-gpu')
        options.add_argument('--log-level=3')
        options.add_argument(f"user-agent={REQUEST_HEADERS['User-Agent']}")
        options.add_argument("--blink-settings=imagesEnabled=false") # Disable images for speed

        driver = uc.Chrome(options=options, use_subprocess=True)
        driver.set_page_load_timeout(60)
        logging.info("Browser launched successfully.")

        # --- Processing Loop ---
        
        # 1. Macro Keywords
        unprocessed_macro = [k for k in MACRO_KEYWORDS if k not in processed_keywords]
        if unprocessed_macro:
            logging.info(f"--- Processing {len(unprocessed_macro)} Macro Keywords ---")
            for keyword in tqdm(unprocessed_macro, desc="Macro Keywords"):
                file_counter = search_and_save_keyword(keyword, True, google_news, visited_urls, file_counter, driver)
                processed_keywords.add(keyword)

        # 2. Ontology Keywords
        unprocessed_csv = [k for k in csv_keywords if k not in processed_keywords]
        if unprocessed_csv:
            logging.info(f"--- Processing {len(unprocessed_csv)} Ontology Keywords ---")
            for keyword in tqdm(unprocessed_csv, desc="Ontology Keywords"):
                file_counter = search_and_save_keyword(keyword, False, google_news, visited_urls, file_counter, driver)
                processed_keywords.add(keyword)

    except Exception as e:
        logging.error(f"Critical Error in Main Loop: {e}")
    finally:
        if driver:
            logging.info("Closing browser...")
            driver.quit()

    logging.info("--- News Crawler Finished ---")

if __name__ == '__main__':
    main()
