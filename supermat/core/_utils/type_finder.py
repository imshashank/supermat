import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson


def find_key_values(data: Any, target_key: str, key_values: dict[str, set] | None = None) -> dict[str, set]:
    """
    Recursively search through nested JSON data to find all instances of a specific key and its values.

    Args:
        data: The JSON data to search through
        target_key: The key to search for
        key_values: Dictionary to store the results (internal use for recursion)

    Returns:
        Dictionary with parent keys and sets of their values
    """
    if key_values is None:
        key_values = defaultdict(set)

    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                # Get the parent key if possible
                parent_key = "root"
                key_values[parent_key].add(str(value))

            # Recurse into nested structures
            find_key_values(value, target_key, key_values)

    elif isinstance(data, list):
        for item in data:
            find_key_values(item, target_key, key_values)

    return key_values


def analyze_json_file(file_path: Path, target_key: str = "type") -> None:
    """
    Analyze a JSON file to find all type values.

    Args:
        file_path: Path to the JSON file
    """
    try:
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())

        type_values = find_key_values(data, target_key)

        print("\nFound 'type' values in the JSON file:")
        print("=====================================")
        for parent, values in type_values.items():
            print(f"\nParent context: {parent}")
            print("Type values:")
            for value in sorted(values):
                print(f"  - {value}")

        print(f"\nTotal unique type values found: {sum(len(values) for values in type_values.values())}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except orjson.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    file_path = Path(sys.argv[1])
    target_key = "type" if len(sys.argv) < 2 else sys.argv[-1]
    analyze_json_file(file_path, target_key)
