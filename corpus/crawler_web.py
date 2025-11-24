import os
import re
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests

# Selenium & Webdriver imports
try:
    import undetected_chromedriver as uc
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.common.exceptions import WebDriverException, TimeoutException
except ImportError as e:
    logging.error(f"Import Error: {e}")
    logging.error("Please ensure you have installed: 'undetected-chromedriver', 'selenium', 'webdriver-manager', 'beautifulsoup4'")
    sys.exit(1)

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'corpus/')
BASE_OUTPUT_DIR = '../data/corpus_raw/web'
OUTPUT_TEXT_DIR = os.path.join(BASE_OUTPUT_DIR, 'txt')
OUTPUT_METADATA_DIR = os.path.join(BASE_OUTPUT_DIR, 'metadata')
OUTPUT_METADATA_FILE = os.path.join(OUTPUT_METADATA_DIR, 'web_corpus_metadata.csv')

# Metadata Headers
METADATA_HEADERS = ['file_id', 'source_url', 'crawl_depth', 'title']

# Seed URLs (Agencies & Missions)
SEED_URLS = [
    # NASA Mars
    'https://mars.nasa.gov/missions/',
    'https://mars.nasa.gov/mars-analogues/',
    'https://solarsystem.nasa.gov/missions/viking-1/in-depth/',
    'https://solarsystem.nasa.gov/missions/mars-pathfinder/in-depth/',
    'https://mars.nasa.gov/mer/mission/science/',
    'https://mars.nasa.gov/mro/mission/science/',
    'https://mars.nasa.gov/msl/mission/science/',
    'https://solarsystem.nasa.gov/missions/maven/in-depth/',
    'https://mars.nasa.gov/mars2020/mission/science/',
    # USGS
    'https://planetarynames.wr.usgs.gov/Page/Rules',
    'https://www.usgs.gov/centers/astrogeology-science-center/science/mars-geologic-mapping',
    # ESA Mars
    'https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/Exploration/Mars_Express',
    'https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/Exploration/ExoMars',
    # General / News / Organizations
    'https://www.planetary.org/space-missions/mars-missions',
    'https://www.astrobio.net/missions-to-mars/',
    'https://www.space.com/mars-exploration',
    # Instrument Teams
    'https://www.uahirise.org/science/',
    'https://www.msl-chemcam.com/',
    'https://an.rsl.wustl.edu/msl/msl-chemcam-team-blogs/',
    'https://www.msss.com/mars-exploration/mars-2020/',
    'https://www.jhuapl.edu/missions/program/mars-exploration',
    # Data Archives
    'https://pds-geosciences.wustl.edu/missions/',
    'https://pds-imaging.jpl.nasa.gov/missions/mars/',
    # International Missions
    'https://www.isro.gov.in/Mars_Orbiter_Mission.html',
    'https://www.planetary.org/space-missions/tianwen-1',
    'https://www.unoosa.org/oosa/en/ourwork/spacetech/missions/emirates-mars-mission.html'
]

# Crawler Parameters
MAX_DEPTH = 1  # 0 = only seed, 1 = seed + 1 link deep
POLITENESS_DELAY = 1.0
REQUEST_TIMEOUT = 30
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
}

# --- Helper Functions ---

def load_existing_data():
    """Loads existing metadata for resume capability."""
    visited_urls = set()
    next_file_counter = 1
    
    # Ensure dirs exist
    os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
    
    file_exists = os.path.exists(OUTPUT_METADATA_FILE)

    if file_exists:
        try:
            logging.info(f"Loading metadata: {OUTPUT_METADATA_FILE}")
            existing_df = pd.read_csv(OUTPUT_METADATA_FILE)
            if not existing_df.empty:
                visited_urls = set(existing_df['source_url'])
                
                # Calculate next ID based on existing file_ids
                # Assumes format 'web_doc_00001'
                numeric_file_ids = pd.to_numeric(
                    existing_df['file_id'].str.extract(r'(\d+)', expand=False),
                    errors='coerce'
                )
                if numeric_file_ids.notna().any():
                    next_file_counter = int(numeric_file_ids.max()) + 1
                
                logging.info(f"Loaded {len(visited_urls)} visited URLs. Next ID: {next_file_counter}")
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            file_exists = False

    if not file_exists:
        pd.DataFrame(columns=METADATA_HEADERS).to_csv(OUTPUT_METADATA_FILE, index=False, encoding='utf-8-sig')
        logging.info("Created new metadata file.")

    return visited_urls, next_file_counter

def is_valid_url(url, base_domain):
    """Checks if URL is valid for crawling (same domain, not a binary file)."""
    try:
        parsed_url = urlparse(url)
        # Ignore common non-text extensions
        if any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.mp4', '.mov', '.gif', '.css', '.js']):
            return False
        # Check valid scheme and domain match
        return bool(parsed_url.scheme) and bool(parsed_url.netloc) and parsed_url.netloc == base_domain
    except:
        return False

def get_clean_text(soup):
    """Extracts and cleans text from BeautifulSoup object."""
    # Remove unwanted tags
    for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        script_or_style.decompose()

    # Try finding main content container
    main_content = soup.find('main') or \
                   soup.find('article') or \
                   soup.find('div', id='content') or \
                   soup.find('div', role='main') or \
                   soup.find('div', class_=re.compile(r'content|main|article|body')) or \
                   soup.body

    text_parts = []
    if main_content:
        # Extract text from specific text-heavy tags
        text_parts = [tag.get_text(separator=' ', strip=True) for tag in
                      main_content.find_all(['p', 'h1', 'h2', 'h3', 'li', 'div', 'span'])]

    # Fallback if main content extraction failed
    if not text_parts:
        text_parts = [tag.get_text(separator=' ', strip=True) for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])]

    # Join and clean up whitespace
    text = ' '.join(part for part in text_parts if part and len(part.split()) > 2)
    text = re.sub(r'\s{2,}', ' ', text)
    return text

# --- Main Execution ---

def main():
    # 1. Setup Directories
    os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)
    logging.info(f"Directories ready: {OUTPUT_TEXT_DIR}")

    # 2. Load State
    visited_urls, file_counter = load_existing_data()

    # 3. Prepare Queue
    # List of tuples: (url, depth)
    # Only add seeds that haven't been visited
    urls_to_crawl = [(url, 0) for url in SEED_URLS if url not in visited_urls]

    if not urls_to_crawl:
        logging.info("All seed URLs processed. To restart, delete the metadata CSV.")
        return

    # 4. Initialize Selenium
    logging.info("Starting Selenium WebDriver...")
    try:
        options = ChromeOptions()
        # options.add_argument('--headless')  # Uncomment for headless
        options.add_argument('--disable-gpu')
        options.add_argument('--log-level=3')
        options.add_argument(f"user-agent={HEADERS['User-Agent']}")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Auto-install driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(REQUEST_TIMEOUT)
        logging.info("WebDriver started successfully.")

    except Exception as e:
        logging.error(f"WebDriver Init Failed: {e}")
        logging.error("Ensure Chrome is installed.")
        return

    # 5. Crawling Loop
    logging.info("--- Starting Web Crawl ---")
    # Using a list as a queue for BFS
    queue = list(urls_to_crawl)
    pbar = tqdm(total=len(queue), desc="Crawling Pages")

    while queue:
        url, current_depth = queue.pop(0)

        if url in visited_urls:
            continue

        pbar.set_description(f"Depth {current_depth}: {url[:40]}...")

        try:
            time.sleep(POLITENESS_DELAY)

            # Navigate
            driver.get(url)
            # Allow JS to render
            time.sleep(3) 

            # Check for soft errors in title
            page_title = driver.title.lower()
            if "403" in page_title or "forbidden" in page_title:
                logging.warning(f"403 Forbidden: {url}")
                visited_urls.add(url)
                continue
            if "404" in page_title or "not found" in page_title:
                logging.warning(f"404 Not Found: {url}")
                visited_urls.add(url)
                continue

            # Extract Content
            page_source = driver.page_source
            visited_urls.add(url) # Mark visited regardless of success to prevent loops

            soup = BeautifulSoup(page_source, 'html.parser')
            clean_text = get_clean_text(soup)

            # Save Data
            if clean_text and len(clean_text) > 100: # Minimum content filter
                # Save Text
                file_name = f"web_doc_{file_counter:05d}.txt"
                file_path = os.path.join(OUTPUT_TEXT_DIR, file_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)

                # Save Metadata
                new_metadata = {
                    'file_id': file_name,
                    'source_url': url,
                    'crawl_depth': current_depth,
                    'title': driver.title.strip() if driver.title else 'N/A'
                }
                pd.DataFrame([new_metadata]).to_csv(OUTPUT_METADATA_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                
                logging.info(f"Saved: {file_name}")
                file_counter += 1
            else:
                logging.warning(f"Skipped {url}: Content empty or too short.")

            # Explore Links (if depth allows)
            if current_depth < MAX_DEPTH:
                base_domain = urlparse(url).netloc
                # Find all links
                links = soup.find_all('a', href=True)
                
                for link in links:
                    # Resolve relative URLs
                    abs_url = urljoin(url, link['href']).split('#')[0] # Remove fragments
                    current_link_domain = urlparse(abs_url).netloc

                    # Validation Logic
                    # 1. Must be valid URL format
                    # 2. Must be same domain (prevent crawling the whole internet)
                    # 3. Must not be visited
                    if is_valid_url(abs_url, current_link_domain):
                        # Check if domain matches seed list domains (optional strictness)
                        # Here we just check if it's same domain as current page
                        if current_link_domain == base_domain:
                            if abs_url not in visited_urls and abs_url not in [q[0] for q in queue]:
                                queue.append((abs_url, current_depth + 1))
                                pbar.total += 1 # Update progress bar total

        except (WebDriverException, TimeoutException) as e:
            logging.error(f"WebDriver Error ({url}): {e}")
            visited_urls.add(url)
        except Exception as e:
            logging.error(f"General Error ({url}): {e}")
            visited_urls.add(url)

        pbar.update(1)

    pbar.close()
    
    logging.info("Closing WebDriver...")
    driver.quit()
    logging.info("--- Web Crawler Finished ---")

if __name__ == '__main__':
    main()
