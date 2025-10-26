import json
import os

def create_small_dataset(original_json_path, new_json_path):
    """
    Creates a smaller dataset split from the original CT-RATE dataset JSON.
    """
    try:
        with open(original_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {original_json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {original_json_path}.")
        return

    if "training" not in data or "validation" not in data:
        print("Error: 'training' or 'validation' keys not found in the original JSON.")
        return

    new_training = data["training"][:3000]
    new_validation = data["validation"][:1000]
    new_test = data["validation"][1000:2000]

    new_dataset = {
        "description": "A smaller subset of the CT-RATE dataset (3k train, 1k val, 1k test).",
        "training": new_training,
        "validation": new_validation,
        "test": new_test
    }

    try:
        with open(new_json_path, 'w') as f:
            json.dump(new_dataset, f, indent=4)
        print(f"Successfully created the small dataset at: {new_json_path}")
        print(f" - Training samples: {len(new_training)}")
        print(f" - Validation samples: {len(new_validation)}")
        print(f" - Test samples: {len(new_test)}")
    except IOError as e:
        print(f"Error writing to file {new_json_path}: {e}")

if __name__ == "__main__":
    # Use absolute paths to ensure correctness
    original_path = r"E:\zizhu\MLLM\VLM_classification\dataset\CT-RATE\CT_RATE_dataset.json"
    new_path = r"E:\zizhu\MLLM\VLM_classification\dataset\CT-RATE\CT_RATE_dataset_small.json"
    create_small_dataset(original_path, new_path)