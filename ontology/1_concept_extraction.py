import os
import time
import json
import logging
import requests
import pandas as pd
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from openai import OpenAI

# --- Configuration ---
# Ideally, load these from environment variables or a .env file
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE") # PLEASE REPLACE OR SET ENV VAR
BASE_URL = "https://api.gptsapi.net/v1" # Custom API endpoint

# Initialize OpenAI Client with timeout handling
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=60.0,
)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to the script location in 'ontology/')
# Adjust these if your data structure changes
PDF_SOURCE_FOLDER = '../data/source_documents/'
OUTPUT_FILE_PATH = '../data/outputs/source_concepts_final.csv'

# Text Processing Parameters
CHUNK_SIZE = 2500
OVERLAP = 300

# --- Knowledge Sources Definition ---
# List of PDFs and URLs to be processed
SOURCES = [
    # Web Resources
    {"name": "IAU_Gazetteer_Nomenclature_Rules", "type": "url", "path": "https://planetarynames.wr.usgs.gov/Page/Rules"},
    {"name": "Viking_Mission", "type": "url", "path": "https://solarsystem.nasa.gov/missions/viking-1/in-depth/"},
    {"name": "Mars_Pathfinder_Mission", "type": "url", "path": "https://solarsystem.nasa.gov/missions/mars-pathfinder/in-depth/"},
    {"name": "MER_Missions_Spirit_Opportunity", "type": "url", "path": "https://mars.nasa.gov/mer/mission/science/"},
    {"name": "MRO_Mission", "type": "url", "path": "https://mars.nasa.gov/mro/mission/science/"},
    {"name": "MSL_Curiosity_Mission", "type": "url", "path": "https://mars.nasa.gov/msl/mission/science/"},
    {"name": "MAVEN_Mission", "type": "url", "path": "https://solarsystem.nasa.gov/missions/maven/in-depth/"},
    {"name": "Perseverance_Mission", "type": "url", "path": "https://mars.nasa.gov/mars2020/mission/science/"},
    
    # Academic Papers / PDFs (Ensure these files exist in PDF_SOURCE_FOLDER)
    {"name": "Planetary Geologic Mapping Protocols.pdf", "type": "pdf", "path": "Planetary Geologic Mapping Protocols.pdf"},
    {"name": "Introduction to Planetary Geomorphology.pdf", "type": "pdf", "path": "Introduction to Planetary Geomorphology_compressed.pdf"},
    {"name": "Terrestrial_Analogs_to_Mars.pdf", "type": "pdf", "path": "Terrestrial_Analogs_to_Mars.pdf"},
    {"name": "Mars_Analogs_Habitability.pdf", "type": "pdf", "path": "Mars analogs Environment, Habitability and Biodiversity.pdf"},
    {"name": "Geology_of_Mars_Evidence.pdf", "type": "pdf", "path": "The Geology of Mars Evidence from Earth-Based Analogs.pdf"},
    {"name": "Evolution_of_Martian_Environment.pdf", "type": "pdf", "path": "Evolution of the Martian Geological Environment and Exploration of Habitability of Mars.pdf"},
]

# --- Helper Functions ---

def extract_text_from_pdf(filename):
    """Extracts text content from a PDF file located in the source folder."""
    file_path = os.path.join(PDF_SOURCE_FOLDER, filename)
    try:
        if not os.path.exists(file_path):
            logging.error(f"PDF file not found: {file_path}")
            return None
            
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        logging.error(f"Error extracting PDF '{filename}': {e}")
        return None

def extract_text_from_url(url):
    """Fetches and parses text content from a URL, removing scripts/styles."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove irrelevant elements
        for script_or_style in soup(["script", "style", "nav", "footer"]):
            script_or_style.decompose()

        # Extract text from specific tags
        text_parts = [tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'div'])]
        text = ' '.join(part.strip() for part in text_parts if part.strip())

        return text
    except requests.RequestException as e:
        logging.error(f"Error fetching URL '{url}': {e}")
        return None

def split_text(text, chunk_size, overlap):
    """Splits text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_concepts_from_gpt(chunk):
    """Calls LLM to extract scientific concepts from a text chunk."""
    prompt = f"""
    You are an expert multi-disciplinary research scientist specializing in comparative planetology.
    Task: Identify core scientific concepts relevant to the comparison between Earth and Mars from the text below.
    
    Output Format: JSON object with a key "concepts" containing a list of objects.
    Each object must have:
    - "concept_name": concise string
    - "description": detailed explanation derived directly from the text
    
    If no concepts found, return {{"concepts": []}}.

    --- TEXT CHUNK ---
    {chunk}
    --- END OF TEXT CHUNK ---
    """
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            if response.choices and response.choices[0].message.content:
                result_json = json.loads(response.choices[0].message.content)
                return result_json.get("concepts", [])
            return []
        except Exception as e:
            logging.error(f"API Call Error (Attempt {attempt + 1}/3): {e}")
            time.sleep(10 * (attempt + 1)) # Exponential backoff
    return []

# --- Main Execution Flow ---

def main():
    """Orchestrates the concept extraction pipeline."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    all_extracted_concepts = []

    for source in SOURCES:
        source_name = source["name"]
        source_type = source["type"]
        source_path = source["path"]

        logging.info(f"--- Processing Source: {source_name} ({source_type}) ---")

        text_content = None
        if source_type == 'pdf':
            text_content = extract_text_from_pdf(source_path)
        elif source_type == 'url':
            text_content = extract_text_from_url(source_path)

        if not text_content:
            logging.warning(f"Skipping {source_name}: No content extracted.")
            continue

        logging.info(f"Extracted {len(text_content)} characters. Splitting and processing...")

        text_chunks = split_text(text_content, CHUNK_SIZE, OVERLAP)

        for i, chunk in enumerate(text_chunks):
            logging.info(f"  -> Processing chunk {i + 1}/{len(text_chunks)}...")
            concepts = get_concepts_from_gpt(chunk)

            if concepts:
                for concept in concepts:
                    concept['source_document'] = source_name
                    all_extracted_concepts.append(concept)

            # Rate limiting precaution
            time.sleep(1) 

    if not all_extracted_concepts:
        logging.warning("No concepts extracted. Please check sources and API.")
        return

    # Save results
    df = pd.DataFrame(all_extracted_concepts)
    # Deduplication based on name and description
    df.drop_duplicates(subset=['concept_name', 'description'], inplace=True)
    
    df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
    logging.info(f"--- Extraction Complete ---")
    logging.info(f"Saved {len(df)} unique concepts to {OUTPUT_FILE_PATH}")

if __name__ == '__main__':
    main()
