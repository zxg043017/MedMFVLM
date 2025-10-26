import json
import argparse
import os
import math
import random

def create_chunks(full_json_path, chunk_size, output_dir):
    """
    Splits a large dataset JSON file into smaller, proportional chunks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(full_json_path, 'r') as f:
        full_data = json.load(f)

    training_data = full_data.get("training", [])
    validation_data = full_data.get("validation", [])
    testing_data = full_data.get("testing", [])

    # Shuffle the training data for better distribution in chunks
    random.shuffle(training_data)

    total_size = len(training_data) + len(validation_data) + len(testing_data)
    if total_size == 0:
        print("The dataset is empty. No chunks will be created.")
        return

    # Calculate proportions
    train_ratio = len(training_data) / total_size
    val_ratio = len(validation_data) / total_size
    test_ratio = len(testing_data) / total_size

    num_chunks = math.ceil(total_size / chunk_size)
    print(f"Total samples: {total_size}")
    print(f"Chunk size: {chunk_size}")
    print(f"Creating {num_chunks} chunks...\n")

    for i in range(num_chunks):
        chunk_data = {"training": [], "validation": [], "testing": []}

        # Calculate number of samples for each split in the current chunk
        train_in_chunk = math.ceil(chunk_size * train_ratio)
        val_in_chunk = math.ceil(chunk_size * val_ratio)
        test_in_chunk = math.ceil(chunk_size * test_ratio)

        # Get training data slice
        start_idx = i * train_in_chunk
        end_idx = start_idx + train_in_chunk
        chunk_data["training"] = training_data[start_idx:end_idx]

        # Get validation data slice
        start_idx = i * val_in_chunk
        end_idx = start_idx + val_in_chunk
        chunk_data["validation"] = validation_data[start_idx:end_idx]

        # Get testing data slice
        start_idx = i * test_in_chunk
        end_idx = start_idx + test_in_chunk
        chunk_data["testing"] = testing_data[start_idx:end_idx]

        # Check if the chunk is empty
        if not any(chunk_data.values()):
            continue

        chunk_filename = os.path.join(output_dir, f"chunk_{i+1:03d}.json")
        with open(chunk_filename, 'w') as f:
            json.dump(chunk_data, f, indent=4)
        
        print(f"Created {chunk_filename} with:")
        print(f"  - {len(chunk_data['training'])} training samples")
        print(f"  - {len(chunk_data['validation'])} validation samples")
        print(f"  - {len(chunk_data['testing'])} testing samples")

    print(f"\nSuccessfully created {num_chunks} JSON chunks in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset JSON file into smaller chunks.")
    parser.add_argument("--full_json_path", type=str, required=True, help="Path to the full dataset JSON file.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of samples per chunk.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the chunked JSON files.")

    args = parser.parse_args()

    create_chunks(args.full_json_path, args.chunk_size, args.output_dir)
