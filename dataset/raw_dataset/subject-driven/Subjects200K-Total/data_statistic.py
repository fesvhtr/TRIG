import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def create_directories(collection_path):
    """Create directories for saving images."""
    os.makedirs(collection_path, exist_ok=True)


def save_image(image_data, output_path):
    """Save image data to a specified path."""
    if image_data:
        img = Image.open(BytesIO(image_data['bytes']))
        img.save(output_path)


def process_row(row, collection_path, subject_id, include_raw_json=False):
    """Process a single row of the DataFrame: save image and return metadata."""
    # Generate subject_id as a zero-padded 6-digit string
    subject_id_str = str(subject_id).zfill(6)

    # Define file paths for saving images
    img_filename = os.path.join(collection_path, f"{subject_id_str}.png")
    
    # Save the image
    save_image(row['image'], img_filename)

    # Collect metadata
    metadata = {
        "subject_id": subject_id_str,
        "img_filename": img_filename,
        "collection": row['collection'],
        "quality_assessment": row['quality_assessment'],
        "description": row['description'],
    }

    # Include raw_json if present in collection2 or collection3
    if include_raw_json:
        metadata["raw_json"] = row.get('raw_json', "")

    return metadata


def load_existing_json(json_output_file):
    """Load existing JSON file if it exists, otherwise return an empty list."""
    if os.path.exists(json_output_file):
        with open(json_output_file, 'r') as f:
            return json.load(f)
    else:
        return []


def get_next_subject_id(existing_json_list):
    """Get the next available subject_id, starting from the max subject_id in the existing list."""
    if existing_json_list:
        # Find the highest existing subject_id
        last_subject_id = max(int(item["subject_id"]) for item in existing_json_list)
        return last_subject_id + 1
    else:
        return 0  # If no existing data, start with 0


def process_parquet_file(parquet_file, collection1_path, collection2_path, collection3_path, 
                          json_output_file1, json_output_file2, json_output_file3, include_raw_json=False):
    """Process a Parquet file and generate JSON output for collection1, collection2, and collection3."""
    # Load DataFrame
    df = pd.read_parquet(parquet_file)
    print(f"Processing {parquet_file}... Columns:", df.dtypes)

    # Prepare JSON lists for each collection
    if not include_raw_json:
        create_directories(collection1_path)
        create_directories(collection2_path)
        json_list1 = load_existing_json(json_output_file1)
        json_list2 = load_existing_json(json_output_file2)
        current_subject_id1 = get_next_subject_id(json_list1)  # Get the next subject_id for collection1
        current_subject_id2 = get_next_subject_id(json_list2)  # Get the next subject_id for collection2
    else:
        create_directories(collection3_path)
        json_list3 = load_existing_json(json_output_file3)
        current_subject_id3 = get_next_subject_id(json_list3)  # Get the next subject_id for collection3

    # Process each row and collect metadata
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if not include_raw_json:
            if row['collection'] == 'collection_1':
                json_data = process_row(row, collection1_path, current_subject_id1, include_raw_json)
                json_list1.append(json_data)
                current_subject_id1 += 1
            elif row['collection'] == 'collection_2':
                json_data = process_row(row, collection2_path, current_subject_id2, include_raw_json)
                json_list2.append(json_data)
                current_subject_id2 += 1
        else:
            json_data = process_row(row, collection3_path, current_subject_id3, include_raw_json)
            json_list3.append(json_data)
            current_subject_id3 += 1

    # Save metadata to JSON files for each collection
    if not include_raw_json:
        with open(json_output_file1, 'w') as f1:
            json.dump(json_list1, f1, indent=4)
        print(f"JSON metadata for collection1 saved to {json_output_file1}")

        with open(json_output_file2, 'w') as f2:
            json.dump(json_list2, f2, indent=4)
        print(f"JSON metadata for collection2 saved to {json_output_file2}")
    else:
        with open(json_output_file3, 'w') as f3:
            json.dump(json_list3, f3, indent=4)
        print(f"JSON metadata for collection3 saved to {json_output_file3}")


if __name__ == "__main__":
    # Define paths and file names
    parquet_dir1 = '../Subjects200K/data'  # For collection1 and collection2
    parquet_dir3 = '../Subjects200K_collection3/data'  # For collection3
    collection1_path = 'collection1'
    collection2_path = 'collection2'
    collection3_path = 'collection3'
    collection1_json_output_file = 'collection1.json'
    collection2_json_output_file = 'collection2.json'
    collection3_json_output_file = 'collection3.json'

    # Process all parquet files for collection1 and collection2 (train-00000-of-00032.parquet to train-00031-of-00032.parquet)
    for i in range(32):
        parquet_file = os.path.join(parquet_dir1, f'train-{i:05d}-of-00032.parquet')
        process_parquet_file(parquet_file, collection1_path, collection2_path, collection3_path,
                              collection1_json_output_file, collection2_json_output_file, collection3_json_output_file, include_raw_json=False)

    # Process all parquet files for collection3 (train-00000-of-00012.parquet to train-00011-of-00012.parquet)
    for i in range(12):
        parquet_file = os.path.join(parquet_dir3, f'train-{i:05d}-of-00012.parquet')
        process_parquet_file(parquet_file, collection1_path, collection2_path, collection3_path,
                              collection1_json_output_file, collection2_json_output_file, collection3_json_output_file, include_raw_json=True)
