import json
import os
import jsonschema
from typing import Dict, Any

def validate_json_schema(schema_file_path: str, data: dict) -> bool:
    """
    Validates a dictionary against a JSON schema file using the jsonschema library.
    Returns True if valid, False otherwise.
    """
    if not schema_file_path:
        # If no schema file path is provided, we skip validation or return True.
        return True
        
    if not os.path.exists(schema_file_path):
        print(f"Warning: Schema file not found at {schema_file_path}. Skipping validation.")
        return True
        
    try:
        with open(schema_file_path, 'r') as f:
            schema = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"JSON Schema Validation Error: {e.message}")
        print(f"Failed path: {list(e.absolute_path)}")
        return False
    except Exception as e:
        print(f"Error during schema validation: {e}")
        return False

def load_json(file_path: str) -> Dict[str, Any]:
    """Loads a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Saves a dict to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)
