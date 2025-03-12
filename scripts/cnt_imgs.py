import os

def traverse_folder(path, indent=0):
    """
    递归遍历指定文件夹，输出每个文件夹中直接包含的文件数量，
    如果该文件夹内还有子文件夹，则继续往里遍历，直到最深层没有子文件夹为止。
    
    参数:
        path: 要遍历的文件夹路径
        indent: 用于控制输出缩进（递归时增加缩进，便于查看层次结构）
    """
    if not os.path.isdir(path):
        print(" " * indent + f"{path} 不是文件夹")
        return

    # 列出该目录下的所有项目
    items = os.listdir(path)
    # 筛选出直接在该目录下的文件和子文件夹
    files = [item for item in items if os.path.isfile(os.path.join(path, item))]
    subfolders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    
    print(" " * indent + f"文件夹: {path} 直接包含 {len(files)} 个文件")
    
    # 递归遍历子文件夹
    for folder in subfolders:
        folder_path = os.path.join(path, folder)
        traverse_folder(folder_path, indent + 4)

# 示例使用：
if __name__ == "__main__":
    root_folder = "/home/muzammal/Projects/TRIG/data/output/p2p_dtm"  # 请替换为你实际的文件夹路径
    traverse_folder(root_folder)

