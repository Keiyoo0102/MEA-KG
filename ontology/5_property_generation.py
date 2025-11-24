import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# --- Configuration ---
# API Settings
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")

# Initialize Client
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=120.0, # Extended timeout for complex generation
)
MODEL_NAME = "gpt-4o-mini"

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'ontology/')
# Input: The structured concept list from Step 4
INPUT_FILE = '../data/outputs/concepts_final_restructured_AI_DRAFT.csv'
# Intermediate Storage: To support resume capability
TEMP_DIR = '../data/outputs/temp_properties/'
# Outputs: Final Relationship and Data Property definitions
REL_OUTPUT_FILE = '../data/outputs/relationships_final_AI_DRAFT.csv'
DATA_OUTPUT_FILE = '../data/outputs/dataproperties_final_AI_DRAFT.csv'

# Batch Processing
BATCH_SIZE = 300

# --- Helper Functions ---

def generate_properties_for_batch(batch_df, batch_num):
    """
    Calls LLM to infer potential Object Properties (Relationships) and Data Properties (Attributes)
    based on a batch of concepts.
    """
    concept_sample_string = batch_df[['concept_name', 'description']].to_string(index=False)
    
    prompt = f"""
    You are a world-class ontology engineer. Your task is to analyze a list of domain concepts and identify the necessary properties to fully describe them in a Knowledge Graph.

    **Input Concepts:**
    {concept_sample_string}

    **Task:**
    1. Identify **Object Properties** (Relationships between concepts). Examples: 'located_in', 'composed_of', 'measured_by'.
    2. Identify **Data Properties** (Attributes with literal values). Examples: 'diameter_km', 'age_mya', 'chemical_formula'.

    **Constraints:**
    - Be comprehensive but avoid redundancy.
    - 'domain' and 'range' should be comma-separated lists of Concept Names (for Object Properties) or Data Types (for Data Properties).
    
    **Output Format:**
    You MUST return a single, valid JSON object with exactly two keys: "object_properties" and "data_properties".
    
    Example JSON Structure:
    {{
        "object_properties": [
            {{ "property_name": "has_composition", "description": "Relates a feature to its material.", "domain": "GeologicFeature", "range": "MaterialComposition" }}
        ],
        "data_properties": [
            {{ "property_name": "elevation", "description": "Height above areoid.", "domain": "LocationRegion", "range": "float" }}
        ]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if content:
            # Validate JSON before saving
            json.loads(content)
            
            # Save to temp file
            file_path = os.path.join(TEMP_DIR, f"batch_{batch_num}_results.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Batch {batch_num} results saved.")
            return True
            
    except json.JSONDecodeError:
        logging.error(f"Batch {batch_num}: Received invalid JSON.")
    except Exception as e:
        logging.error(f"Error processing batch {batch_num}: {e}")
        
    return False

def synthesize_definitions(property_name, definitions, prop_type):
    """
    Synthesizes multiple definitions of the same property into one definitive version using LLM.
    """
    definitions_str = "\n---\n".join(
        [f"Description: {d.get('description', 'N/A')}\nDomain: {d.get('domain', 'N/A')}\nRange: {d.get('range', 'N/A')}"
         for d in definitions]
    )
    
    range_instruction = 'The `range` must be a data type like "str", "int", "float", or "bool".' if prop_type == 'data' else 'The `range` should be a concept type like "MesoScale" or "Geological Process".'

    prompt = f"""
    You are a senior ontology editor. You have received multiple suggested definitions for the property "{property_name}".
    Your task is to synthesize them into a SINGLE, definitive, high-quality definition.

    **Suggestions:**
    {definitions_str}

    **Goal:**
    Create the single best definition with three keys: "description", "domain", and "range".
    - `description`: Clear and comprehensive.
    - `domain`: Accurate, comma-separated list of domain concepts.
    - `range`: Accurate, comma-separated list for the range. {range_instruction}

    **Output:**
    A single valid JSON object.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if content:
            return json.loads(content)
    except Exception as e:
        logging.error(f"Error synthesizing '{property_name}': {e}")
    
    return None

# --- Main Execution Flow ---

def main():
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REL_OUTPUT_FILE), exist_ok=True)

    # --- Phase 1: Batch Generation ---
    logging.info("--- Phase 1: Generating Properties from Concepts ---")
    
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
        logging.error("Please run 'ontology/4_structure_generation.py' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    num_batches = (len(df) // BATCH_SIZE) + (1 if len(df) % BATCH_SIZE > 0 else 0)

    # Check for existing batches (Resume Logic)
    all_batches_exist = True
    for i in range(num_batches):
        if not os.path.exists(os.path.join(TEMP_DIR, f"batch_{i + 1}_results.json")):
            all_batches_exist = False
            break

    if all_batches_exist:
        logging.info("All batch files exist. Skipping generation phase.")
    else:
        for i in range(num_batches):
            if os.path.exists(os.path.join(TEMP_DIR, f"batch_{i + 1}_results.json")):
                logging.info(f"Batch {i + 1}/{num_batches} exists. Skipping.")
                continue
            
            logging.info(f"Processing Batch {i + 1}/{num_batches}...")
            start_index = i * BATCH_SIZE
            end_index = start_index + BATCH_SIZE
            batch_df = df.iloc[start_index:end_index]
            
            success = generate_properties_for_batch(batch_df, i + 1)
            if not success:
                logging.warning(f"Batch {i + 1} failed. You may need to re-run.")
            
            time.sleep(2) # Rate limiting

    # --- Phase 2: Synthesis & Aggregation ---
    logging.info("--- Phase 2: Synthesizing & Aggregating Definitions ---")

    all_object_props = {}
    all_data_props = {}

    # Load all temp files
    for filename in os.listdir(TEMP_DIR):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(TEMP_DIR, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for prop in data.get("object_properties", []):
                        name = prop.get("property_name")
                        if name:
                            if name not in all_object_props: all_object_props[name] = []
                            all_object_props[name].append(prop)
                            
                    for prop in data.get("data_properties", []):
                        name = prop.get("property_name")
                        if name:
                            if name not in all_data_props: all_data_props[name] = []
                            all_data_props[name].append(prop)
            except json.JSONDecodeError:
                logging.error(f"Skipping corrupted file: {filename}")

    # Synthesize Object Properties
    if not os.path.exists(REL_OUTPUT_FILE):
        final_object_props = []
        for name, definitions in tqdm(all_object_props.items(), desc="Synthesizing Relations"):
            synthesized = synthesize_definitions(name, definitions, 'object')
            if synthesized:
                synthesized['property_name'] = name
                final_object_props.append(synthesized)
            time.sleep(0.5)
        
        if final_object_props:
            rel_df = pd.DataFrame(final_object_props)[['property_name', 'description', 'domain', 'range']]
            rel_df.to_csv(REL_OUTPUT_FILE, index=False, encoding='utf-8-sig')
            logging.info(f"Saved {len(rel_df)} Object Properties to {REL_OUTPUT_FILE}")
    else:
        logging.info(f"File exists: {REL_OUTPUT_FILE}. Skipping synthesis.")

    # Synthesize Data Properties
    if not os.path.exists(DATA_OUTPUT_FILE):
        final_data_props = []
        for name, definitions in tqdm(all_data_props.items(), desc="Synthesizing Attributes"):
            synthesized = synthesize_definitions(name, definitions, 'data')
            if synthesized:
                synthesized['property_name'] = name
                final_data_props.append(synthesized)
            time.sleep(0.5)
        
        if final_data_props:
            data_df = pd.DataFrame(final_data_props)[['property_name', 'description', 'domain', 'range']]
            data_df.to_csv(DATA_OUTPUT_FILE, index=False, encoding='utf-8-sig')
            logging.info(f"Saved {len(data_df)} Data Properties to {DATA_OUTPUT_FILE}")
    else:
        logging.info(f"File exists: {DATA_OUTPUT_FILE}. Skipping synthesis.")

    logging.info(f"--- Property Generation Workflow Complete ---")

if __name__ == '__main__':
    main()
