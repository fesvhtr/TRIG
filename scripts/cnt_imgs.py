import os

def count_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        raise ValueError(f"not exist {directory_path}")

    if not os.path.isdir(directory_path):
        raise ValueError(f"err path {directory_path}")

    return len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

folder_path = "/home/muzammal/Projects/TRIG/data/output/p2p/freediff"
file_count = count_files_in_directory(folder_path)
print(f"'{folder_path}' : {file_count}")
