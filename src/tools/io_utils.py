import json

from typing import Dict, Any


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    ### Args:
        `file_path (str)`: Path to the JSON file.

    ### Returns:
        `Dict[str, Any]`: Contents of the JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)
