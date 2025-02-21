import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def create_directories(src_path, edited_path):
    """Create directories for saving source and edited images."""
    os.makedirs(src_path, exist_ok=True)
    os.makedirs(edited_path, exist_ok=True)


def save_image(image_data, output_path):
    """Save image data to a specified path."""
    if image_data:
        img = Image.open(BytesIO(image_data['bytes']))
        img.save(output_path)


def inspect_parquet_dtypes(parquet_file):
    """Inspect the dtypes of columns in the parquet file."""
    df = pd.read_parquet(parquet_file)
    print("Data types in the Parquet file:")
    print(df.dtypes)
    return df.dtypes


def process_row(row, src_path, edited_path):
    """Process a single row of the DataFrame: save images and return metadata."""
    omni_edit_id = row['omni_edit_id']

    # Define file paths
    src_img_filename = os.path.join(src_path, f"{omni_edit_id}.png")
    edited_img_filename = os.path.join(edited_path, f"{omni_edit_id}.png")

    # Save source and edited images
    save_image(row['src_img'], src_img_filename)
    save_image(row['edited_img'], edited_img_filename)

    # Return metadata as a dictionary
    metadata = {
        "omni_edit_id": omni_edit_id,
        "task": row['task'],
        "edited_prompt_list": row['edited_prompt_list'],
        "width": row['width'],
        "height": row['height'],
        "sc_score_1": row['sc_score_1'],
        "sc_score_2": row['sc_score_2'],
        "sc_reasoning": row['sc_reasoning'],
        "pq_score": row['pq_score'],
        "pq_reasoning": row['pq_reasoning'],
        "o_score": row['o_score'],
        "src_img_filename": src_img_filename,
        "edited_img_filename": edited_img_filename
    }

    # Check for NumPy arrays and convert to list if necessary
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()

    return metadata


def process_parquet_file(parquet_file, src_path, edited_path, json_output_file):
    """Main function to process a Parquet file and generate JSON output."""
    # Load DataFrame
    df = pd.read_parquet(parquet_file)
    print("DataFrame loaded. Columns:", df.dtypes)

    # Create directories
    create_directories(src_path, edited_path)

    # Process each row and collect metadata
    json_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        json_data = process_row(row, src_path, edited_path)
        json_list.append(json_data)

    # Save metadata to JSON file
    with open(json_output_file, 'w', encoding='utf-8') as f:
        for json_data in json_list:
            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')

    print(f"JSON metadata saved to {json_output_file}")


if __name__ == "__main__":
    # Define paths and file names
    parquet_file = r'H:\ProjectsPro\TRIG\dataset\raw_dataset\image-editing\dev-00000-of-00001.parquet'
    src_path = 'dev/src'
    edited_path = 'dev/edited'
    json_output_file = 'prompt_dev.jsonl'

    # Inspect dtypes in the Parquet file
    inspect_parquet_dtypes(parquet_file)

    # Process the Parquet file
    process_parquet_file(parquet_file, src_path, edited_path, json_output_file)
