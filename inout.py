from typing import List, Dict, Any
import json

FILE_PATH = "data.json"


def append_to_json(new_data: List[Dict[str, Any]], file_path: str = FILE_PATH) -> None:
    existing_data = load_from_json(file_path)
    existing_data.extend(new_data)
    save_to_json(existing_data, file_path)


def save_to_json(data: List[Dict[str, Any]], file_path: str = FILE_PATH) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_from_json(file_path: str = FILE_PATH) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def extract_from_json(edge_no: int, coord_no: int, seed: int, replica_no: int, file_path: str = FILE_PATH):
    data = load_from_json(file_path)
    return next((d for d in data if
                 d.get("edge_no") == edge_no and d.get("coord_no") == coord_no and d.get("seed") == seed and d.get(
                     "replica_no") == replica_no), None)


def extract_key_values(json_file, key):
    with open(json_file, 'r') as file:
        data = json.load(file)

    if key == "pop":
        return [[item for sublist in obj.get(key, []) for item in sublist] for obj in data]
    else:
        return [obj.get(key, []) for obj in data]


def exists_in_json(edge_no: int, coord_no: int, seed: int, replica_no: int, file_path: str = FILE_PATH) -> bool:
    data = load_from_json(file_path)
    return any(d.get("edge_no") == edge_no and d.get("coord_no") == coord_no and d.get("seed") == seed and d.get(
        "replica_no") == replica_no for d in data)
