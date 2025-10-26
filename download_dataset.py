import json
import os
import argparse
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_files_from_json(json_path, split_name, repo_id, hf_token):
    """
    Downloads files for a specific split from a dataset JSON manifest.

    Args:
        json_path (str): Path to the dataset JSON file (e.g., 'CT_RATE_dataset_small.json').
        split_name (str): The split to download ('training', 'validation', 'test').
        repo_id (str): The Hugging Face repository ID.
        hf_token (str): Your Hugging Face API token.
    """
    # --- 1. Load the JSON file ---
    try:
        with open(json_path, 'r') as f:
            dataset_manifest = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing JSON file {json_path}: {e}")
        return

    if split_name not in dataset_manifest:
        print(f"Error: Split '{split_name}' not found in {json_path}. Available splits: {list(dataset_manifest.keys())}")
        return

    file_list = dataset_manifest[split_name]
    print(f"Found {len(file_list)} files to download for the '{split_name}' split.")

    # --- 2. Loop through files and download ---
    for file_entry in tqdm(file_list, desc=f"Downloading {split_name} split"):
        # e.g., hf_path = "./train_data/images/train_1_a_1.nii.gz"
        hf_path = file_entry.get("img_t1")
        if not hf_path:
            continue

        # Remove the leading "./" if it exists
        if hf_path.startswith('./'):
            hf_path = hf_path[2:]

        # Parse the path to get subfolder and filename
        # subfolder = "train_data/images", filename = "train_1_a_1.nii.gz"
        subfolder = os.path.dirname(hf_path)
        filename = os.path.basename(hf_path)
        
        # Define where the file should be saved locally
        # This will save to "E:\zizhu\MLLM\VLM_classification\dataset\CT-RATE\train_data\images\..."
        local_dir = os.path.dirname(json_path)

        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                token=hf_token,
                subfolder=subfolder,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as e:
            print(f"Failed to download {filename}. Error: {e}")

    print(f"Finished downloading the '{split_name}' split.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specific splits from the CT-RATE dataset on Hugging Face.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="dataset/CT-RATE/CT_RATE_dataset_small.json",
        help="Path to the dataset JSON manifest file."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=['training', 'validation', 'test'],
        help="The dataset split to download ('training', 'validation', or 'test')."
    )
    args = parser.parse_args()

    REPO_ID = 'ibrahimhamamci/CT-RATE'
    # IMPORTANT: Set the HUGGING_FACE_TOKEN environment variable before running the script
    # For example: export HUGGING_FACE_TOKEN='your_token_here'
    HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")

    # Construct the absolute path for the JSON file
    # Assumes the script is run from the project root 'VLM_classification'
    abs_json_path = os.path.join(os.getcwd(), args.json_path)

    download_files_from_json(abs_json_path, args.split, REPO_ID, HF_TOKEN)
