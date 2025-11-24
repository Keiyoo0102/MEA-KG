import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# --- Configuration ---
# API Settings
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # Replace or set env var
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")

# Initialize Client
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=60.0,
)
MODEL_NAME = "gpt-4o"

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to the script location in 'ontology/')
# NOTE: Ensure 'concepts_final_cleaned.csv' exists. This should be the file after you manually reviewed the concepts.
INPUT_FILE = '../data/outputs/concepts_final_cleaned.csv'
OUTPUT_CSV = '../data/outputs/concepts_final_restructured_AI_DRAFT.csv'

# --- Helper Functions ---

def refine_and_structure_concept(concept_name, concept_desc, all_concept_names):
    """
    Calls LLM to disambiguate a concept and place it within a multi-scale hierarchy (Macro/Meso/Micro).
    """
    # Provide context to the LLM by showing a sample of other concepts
    context_concepts_sample = ", ".join(all_concept_names[:100])

    prompt = f"""
    You are a world-class planetary science ontologist. Your primary goal is to disambiguate potentially vague concepts and then place them precisely within a multi-scale hierarchy. This is a two-step process.

    **Step 1: Refine the Concept Name.**
    Analyze the original concept name and its description. If the name is ambiguous (e.g., "Mineral", "Atmosphere"), create a new, more specific and unambiguous `refined_concept_name`. Use modifiers like "Distribution", "Composition", "Process", "Feature", "Layer", "Model", etc. If the original name is already specific and clear (e.g., "Gale Crater"), use it as the refined name.

    **Step 2: Structure the Refined Concept.**
    Based on the **newly refined concept name** and the original description, determine its position in the ontology.
    The top-level parents MUST be one of three scales: MacroScale, MesoScale, or MicroScale.
    - MacroScale: For planetary-scale entities (e.g., Global Geologic Map, Southern Highlands, Amazonian Period, Atmospheric Composition).
    - MesoScale: For regional-scale entities (e.g., Gale Crater, Alluvial Fan, Fluvial Process, Rover Mission).
    - MicroScale: For in-situ, compositional, or microscopic entities (e.g., Mineral Composition, Jarosite, ChemCam Instrument, Rock Texture).

    Here is a small sample of other concepts for context: {context_concepts_sample}

    **Target Concept to Process:**
    - Original Name: "{concept_name}"
    - Original Description: "{concept_desc}"

    You MUST return a single JSON object with exactly THREE keys:
    1. "refined_concept_name": Your new, specific, and unambiguous name for the concept.
    2. "scale": The top-level scale for the *refined* concept. Must be one of ["MacroScale", "MesoScale", "MicroScale"].
    3. "parent_concept": The immediate parent for the *refined* concept. This could be a scale name (e.g., "MesoScale") or a more specific concept (e.g., "Impact Crater").

    **Example Walkthroughs:**
    - "Mineral" (Global map description) -> {{"refined_concept_name": "Global Mineral Distribution", "scale": "MacroScale", "parent_concept": "MacroScale"}}
    - "Mineral" (Rock composition description) -> {{"refined_concept_name": "Mineral Composition", "scale": "MicroScale", "parent_concept": "MicroScale"}}
    - "Gale Crater" -> {{"refined_concept_name": "Gale Crater", "scale": "MesoScale", "parent_concept": "Impact Crater"}}
    """
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            return None
        except json.JSONDecodeError:
             logging.error(f"JSON Decode Error for '{concept_name}'")
        except Exception as e:
            logging.error(f"API call failed for '{concept_name}' (attempt {attempt + 1}/3): {e}")
            time.sleep(5 * (attempt + 1))
            
    return None

# --- Main Execution Flow ---

def main():
    # 1. Check Input
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
        logging.error("Please ensure you have a cleaned concept list named 'concepts_final_cleaned.csv' in data/outputs/.")
        # Fallback check: if cleaned file doesn't exist, maybe use the raw extraction?
        # INPUT_FILE = '../data/outputs/source_concepts_final.csv' 
        return

    logging.info(f"Loading cleaned concepts from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return

    df.dropna(subset=['description'], inplace=True)
    all_names = df['concept_name'].tolist()
    
    logging.info(f"Starting structure generation for {len(df)} concepts...")
    
    results = []
    # Use tqdm for progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Refining Ontology"):
        structure_info = refine_and_structure_concept(row['concept_name'], row['description'], all_names)

        row_dict = row.to_dict()
        if structure_info:
            row_dict['refined_concept_name'] = structure_info.get('refined_concept_name')
            row_dict['scale'] = structure_info.get('scale')
            row_dict['parent_concept_name'] = structure_info.get('parent_concept')
        else:
            row_dict['refined_concept_name'] = row['concept_name']  # Fallback
            row_dict['scale'] = "Unclassified"
            row_dict['parent_concept_name'] = "Unclassified"
        
        results.append(row_dict)
        # Rate limiting precaution
        time.sleep(1)

    structured_df = pd.DataFrame(results)

    # Post-processing: Build Hierarchy IDs
    # Use refined name as the primary name
    structured_df['concept_name'] = structured_df['refined_concept_name']
    
    # Assign IDs (starting from 3, as 0,1,2 are reserved for Scales)
    structured_df['concept_id'] = structured_df.index + 3
    
    # Create a mapping from Name -> ID for parent linking
    name_to_id = pd.Series(structured_df.concept_id.values, index=structured_df.concept_name).to_dict()
    
    # Add Root Nodes manually
    name_to_id['MacroScale'] = 0
    name_to_id['MesoScale'] = 1
    name_to_id['MicroScale'] = 2

    # Map parent names to parent IDs
    structured_df['parent_concept_id'] = structured_df['parent_concept_name'].map(name_to_id)
    
    # If parent specific concept not found (e.g. parent is just "MesoScale"), map to scale ID
    structured_df.loc[structured_df['parent_concept_id'].isnull(), 'parent_concept_id'] = structured_df['scale'].map(name_to_id)

    # Create Root Node DataFrame
    root_nodes = pd.DataFrame([
        {'concept_id': 0, 'concept_name': 'MacroScale', 'description': 'Planetary-scale concepts', 'parent_concept_id': -1},
        {'concept_id': 1, 'concept_name': 'MesoScale', 'description': 'Regional-scale concepts', 'parent_concept_id': -1},
        {'concept_id': 2, 'concept_name': 'MicroScale', 'description': 'Local or compositional concepts', 'parent_concept_id': -1},
    ])
    
    # Combine Roots and Concepts
    final_df = pd.concat([root_nodes, structured_df], ignore_index=True)

    # Final Formatting
    # Fill NaN parents with -2 (Orphan)
    final_df['parent_concept_id'] = final_df['parent_concept_id'].fillna(-2).astype(int)
    
    # Select columns
    cols_to_keep = ['concept_id', 'concept_name', 'description', 'parent_concept_id']
    final_df = final_df[cols_to_keep]

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    logging.info(f"--- Structure Generation Complete ---")
    logging.info(f"AI-generated structure saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
