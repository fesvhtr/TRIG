import os
import pandas as pd


def process_and_split_parquet(input_dir, output_dir):
    """Process and split Parquet files based on collection value."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    collection1_list = []
    collection2_list = []
    
    for i in range(32):
        input_file = os.path.join(input_dir, f'train-000{i:02d}-of-00032.parquet')
        
        print(f"Processing {input_file}...")
        df = pd.read_parquet(input_file)
        
        collection1_df = df[df['collection'] == 'collection_1']
        collection2_df = df[df['collection'] == 'collection_2']
        
        collection1_list.append(collection1_df)
        collection2_list.append(collection2_df)
    
    collection1_combined = pd.concat(collection1_list, ignore_index=True)
    collection2_combined = pd.concat(collection2_list, ignore_index=True)
    
    collection1_output_file = os.path.join(output_dir, 'collection1.parquet')
    collection2_output_file = os.path.join(output_dir, 'collection2.parquet')
    
    collection1_combined.to_parquet(collection1_output_file)
    collection2_combined.to_parquet(collection2_output_file)
    
    print(f"Processing complete. Files saved to {collection1_output_file} and {collection2_output_file}")


def process_and_merge_parquet(input_dir, output_dir):
    """Merge Parquet files in batches and reorder columns."""
    
    column_order = [
        'collection', 
        'image', 
        'quality_assessment',
        'description',
        'raw_json'
    ]
    
    data_list_batch1 = []
    data_list_batch2 = []

    for i in range(6):
        input_file = os.path.join(input_dir, f'train-000{i:02d}-of-00012.parquet')
        print(f"Processing {input_file}...")
        df = pd.read_parquet(input_file)
        df = df[column_order]
        data_list_batch1.append(df)
    
    batch1_df = pd.concat(data_list_batch1, ignore_index=True)
    batch1_file = os.path.join(output_dir, 'collection3_1.parquet')
    batch1_df.to_parquet(batch1_file)
    print(f"Batch 1 merged. Shape: {batch1_df.shape}")
    
    for i in range(6, 12):
        input_file = os.path.join(input_dir, f'train-000{i:02d}-of-00012.parquet')
        print(f"Processing {input_file}...")
        df = pd.read_parquet(input_file)
        df = df[column_order]
        data_list_batch2.append(df)
    
    batch2_df = pd.concat(data_list_batch2, ignore_index=True)
    batch2_file = os.path.join(output_dir, 'collection3_2.parquet')
    batch2_df.to_parquet(batch2_file)
    print(f"Batch 2 merged. Shape: {batch2_df.shape}")


if __name__ == "__main__":
    input_collection12_dir = '../Subjects200K/data'
    input_collection3_dir = '../Subjects200K_collection3/data'
    output_dir = './'
    
    process_and_split_parquet(input_collection12_dir, output_dir)
    # process_and_merge_parquet(input_collection3_dir, output_dir)
