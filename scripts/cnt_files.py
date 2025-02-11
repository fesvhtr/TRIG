import os

def count_files_in_directory(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"'{directory}' 不是一个有效的文件夹路径")

    return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))

def count_and_delete_files(directory):
    """查找并删除所有以 'TA-R' 或 'TA-S' 开头的文件"""
    if not os.path.isdir(directory):
        raise ValueError(f"'{directory}' 不是一个有效的文件夹路径")

    files_to_delete = [f for f in os.listdir(directory) if f.startswith('TA-R_TA-S') and os.path.isfile(os.path.join(directory, f))]
    file_count = len(files_to_delete)

    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        try:
            os.remove(file_path)
            print(f"已删除：{file_path}")
        except Exception as e:
            print(f"删除 {file_path} 失败: {e}")

    print(f"总共找到并删除 {file_count} 个文件")

# 示例：修改为你的目标文件夹路径
folder_path = "/home/zsc/TRIG/data/output/t2i/pixart_sigma"
file_count = count_files_in_directory(folder_path)
print(f"文件夹 '{folder_path}' 中的文件数：{file_count}")
count_and_delete_files(folder_path)