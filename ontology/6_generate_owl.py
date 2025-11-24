import os
import re
import logging
import pandas as pd
from owlready2 import *

# --- Configuration ---
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths (Relative to 'ontology/')
# Inputs: Using files generated from previous steps (Step 4 and Step 5)
CONCEPTS_FILE = '../data/outputs/concepts_final_restructured_AI_DRAFT.csv'
RELATIONS_FILE = '../data/outputs/relationships_final_AI_DRAFT.csv'
DATAPROPS_FILE = '../data/outputs/dataproperties_final_AI_DRAFT.csv'

# Output: Final Ontology File
ONTOLOGY_OUTPUT_FILE = '../data/ontology/MEA_Ontology.owl'

# Ontology Base IRI
ONTOLOGY_IRI = "http://www.mea-kg.com/ontology.owl"

# Data Type Mapping
DATATYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "bool": bool,
    "boolean": bool
}

# --- Helper Functions ---

def sanitize_name(name):
    """Sanitize concept names to be valid OWL identifiers."""
    if pd.isna(name): return "Unnamed"
    # Replace non-alphanumeric characters with underscores
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

# --- Main Execution Flow ---

def main():
    logging.info("--- Starting Automated OWL Ontology Generation ---")

    # 1. Check Inputs
    if not os.path.exists(CONCEPTS_FILE):
        logging.error(f"Concepts file not found: {CONCEPTS_FILE}")
        logging.error("Please run 'ontology/4_structure_generation.py' first.")
        return

    # 2. Initialize Ontology
    # owlready2 uses 'file://' syntax implicitly when saving, but here we set the logical IRI
    onto = get_ontology(ONTOLOGY_IRI)

    logging.info(f"Loading concepts from {CONCEPTS_FILE}...")
    concepts_df = pd.read_csv(CONCEPTS_FILE)
    
    # Dictionary to store created Owlready2 classes: {concept_id: ClassObject}
    created_classes = {}

    with onto:
        # 3. Create Classes and Hierarchy
        # We need to process parents first, so we sort by depth.
        # If depth isn't pre-calculated, we calculate it recursively.
        
        id_to_row = {row['concept_id']: row.to_dict() for index, row in concepts_df.iterrows()}
        memo = {}

        def get_depth(concept_id):
            if concept_id in memo: return memo[concept_id]
            # Root or orphan nodes
            if concept_id <= 0: return 0 
            if concept_id not in id_to_row: return 0
            
            parent_id = id_to_row[concept_id]['parent_concept_id']
            # Prevent infinite loops if circular reference exists (simple check)
            if parent_id == concept_id: return 0
            
            depth = get_depth(parent_id) + 1
            memo[concept_id] = depth
            return depth

        logging.info("Calculating hierarchy depth...")
        concepts_df['depth'] = concepts_df['concept_id'].apply(get_depth)
        sorted_concepts_df = concepts_df.sort_values(by='depth')

        logging.info("Creating OWL Classes...")
        for index, row in sorted_concepts_df.iterrows():
            concept_name_raw = row.get('concept_name', row.get('refined_concept_name'))
            class_name = sanitize_name(concept_name_raw)
            description = str(row.get('description', ''))
            parent_id = int(row.get('parent_concept_id', -2))

            # Determine Parent Class
            parent_class_list = [Thing] # Default parent is owl:Thing
            if parent_id in created_classes:
                parent_class_list = [created_classes[parent_id]]
            
            # Create the class dynamically
            new_class = types.new_class(class_name, tuple(parent_class_list))
            new_class.comment = [description]
            
            # Store for children to use
            created_classes[row['concept_id']] = new_class

        logging.info(f"Created {len(created_classes)} classes.")

        # 4. Create Object Properties (Relationships)
        if os.path.exists(RELATIONS_FILE):
            logging.info(f"Loading relationships from {RELATIONS_FILE}...")
            relations_df = pd.read_csv(RELATIONS_FILE)
            
            # Map cleaned names back to OWL classes
            # We use the class name as key because external files use names, not IDs
            name_to_class = {cls.name: cls for cls in created_classes.values()}

            for index, row in relations_df.iterrows():
                prop_name = sanitize_name(row['property_name'])
                description = str(row.get('description', ''))
                
                # Create Property
                new_prop = types.new_class(prop_name, (ObjectProperty,))
                new_prop.comment = [description]

                # Set Domain
                domain_names = [sanitize_name(c.strip()) for c in str(row.get('domain', '')).split(',')]
                domain_classes = [name_to_class[n] for n in domain_names if n in name_to_class]
                if domain_classes:
                    new_prop.domain = domain_classes

                # Set Range
                range_names = [sanitize_name(c.strip()) for c in str(row.get('range', '')).split(',')]
                range_classes = [name_to_class[n] for n in range_names if n in name_to_class]
                if range_classes:
                    new_prop.range = range_classes
        else:
            logging.warning(f"Relationships file not found: {RELATIONS_FILE}. Skipping.")

        # 5. Create Data Properties (Attributes)
        if os.path.exists(DATAPROPS_FILE):
            logging.info(f"Loading data properties from {DATAPROPS_FILE}...")
            dataprops_df = pd.read_csv(DATAPROPS_FILE)
            name_to_class = {cls.name: cls for cls in created_classes.values()}

            for index, row in dataprops_df.iterrows():
                prop_name = sanitize_name(row['property_name'])
                description = str(row.get('description', ''))

                # Create Property
                new_prop = types.new_class(prop_name, (DataProperty,))
                new_prop.comment = [description]

                # Set Domain
                domain_names = [sanitize_name(c.strip()) for c in str(row.get('domain', '')).split(',')]
                domain_classes = [name_to_class[n] for n in domain_names if n in name_to_class]
                if domain_classes:
                    new_prop.domain = domain_classes

                # Set Range (Data Type)
                range_str_raw = str(row.get('range', 'str')).lower().strip()
                range_type = DATATYPE_MAP.get(range_str_raw, str)
                new_prop.range = [range_type]
        else:
            logging.warning(f"Data properties file not found: {DATAPROPS_FILE}. Skipping.")

    # 6. Save Ontology
    logging.info("Saving OWL file...")
    os.makedirs(os.path.dirname(ONTOLOGY_OUTPUT_FILE), exist_ok=True)
    onto.save(file=ONTOLOGY_OUTPUT_FILE, format="rdfxml")

    logging.info(f"--- Ontology Generation Complete ---")
    logging.info(f"OWL file saved to: {ONTOLOGY_OUTPUT_FILE}")

if __name__ == '__main__':
    main()
