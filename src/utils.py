from pathlib import Path
import json


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, file_path):
    """Save a Python dictionary as a JSON file."""
    file_path = Path(file_path)

    ensure_directory(file_path.parent)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def load_json(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as file:
        return json.load(file)