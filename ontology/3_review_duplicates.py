import os
import time
import json
import logging
import pandas as pd
from openai import OpenAI

# --- Configuration ---
# API Settings
# Ideally, load these from environment variables
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # Replace with your key or set env var
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1") # Custom endpoint if needed

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
# Input comes from step 2 output
INPUT_FILE_PATH = '../data/outputs/concepts_to_review.csv'
OUTPUT_FILE_PATH = '../data/outputs/concepts_review_ai_suggestions.csv'

# --- Helper Functions ---

def get_review_decision(concept1_name, concept1_desc, concept2_name, concept2_desc):
    """
    Calls LLM to decide whether two concepts should be merged, kept separate, or restructured.
    """
    prompt = f"""
    You are a meticulous planetary scientist and ontologist, an expert in creating clear and non-redundant knowledge structures.
    Your task is to analyze the following pair of scientific concepts, which have been flagged as having highly similar descriptions.
    
    Based on their names and descriptions, you must decide if they represent the same concept and should be merged, or if they are distinct enough to be kept separate.

    Concept 1 Name: "{concept1_name}"
    Concept 1 Description: "{concept1_desc}"

    Concept 2 Name: "{concept2_name}"
    Concept 2 Description: "{concept2_desc}"

    You MUST choose one of three options for your decision:
    1. "Merge": If they are essentially the same concept, possibly with slightly different wording.
    2. "Keep": If they represent distinct, important concepts that should both exist in the ontology, despite description similarity.
    3. "Restructure": If they are related but not identical, for example, one is a part of the other or a sub-type of the other.

    You MUST return your answer as a single JSON object with exactly two keys:
    - "decision": Your choice, which must be one of ["Merge", "Keep", "Restructure"].
    - "reason": A brief, one-sentence justification for your decision.

    Example JSON output:
    {{
      "decision": "Merge",
      "reason": "Both concepts describe the same fan-shaped sedimentary deposit, with 'Alluvial Fan' being the more standard term."
    }}
    """
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                result_json = json.loads(content)
                return result_json
            return None
            
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response (Attempt {attempt + 1}/3)")
        except Exception as e:
            logging.error(f"API call failed (Attempt {attempt + 1}/3): {e}")
            time.sleep(5 * (attempt + 1)) # Exponential backoff
            
    return None

# --- Main Execution Flow ---

def main():
    """
    Main function for AI-assisted Duplicate Review.
    """
    # 1. Check Input
    if not os.path.exists(INPUT_FILE_PATH):
        logging.error(f"Input file not found: {INPUT_FILE_PATH}")
        logging.error("Please run 'ontology/2_concept_alignment.py' first.")
        return

    logging.info(f"Loading concepts to review from {INPUT_FILE_PATH}...")
    try:
        df = pd.read_csv(INPUT_FILE_PATH)
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return

    if df.empty:
        logging.warning("Input file is empty. No concepts to review.")
        return

    results = []
    total_pairs = len(df)
    logging.info(f"Found {total_pairs} pairs to review.")

    # 2. Process each pair
    for index, row in df.iterrows():
        logging.info(f"Processing pair {index + 1}/{total_pairs}: '{row['concept_name_1']}' vs '{row['concept_name_2']}'")

        suggestion = get_review_decision(
            row['concept_name_1'], row['description_1'],
            row['concept_name_2'], row['description_2']
        )

        if suggestion:
            result_row = row.to_dict()
            result_row['ai_decision'] = suggestion.get('decision')
            result_row['ai_reason'] = suggestion.get('reason')
            results.append(result_row)
        else:
            logging.warning(f"Skipping pair {index + 1} due to AI failure.")

        # Rate limiting precaution
        time.sleep(1)

    # 3. Save Results
    if results:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        
        review_df = pd.DataFrame(results)
        review_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
        
        logging.info(f"--- AI Review Complete ---")
        logging.info(f"AI suggestions saved to {OUTPUT_FILE_PATH}")
    else:
        logging.warning("No results generated.")

if __name__ == '__main__':
    main()
