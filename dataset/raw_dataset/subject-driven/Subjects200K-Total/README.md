# Subjects200K
---
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: image
    dtype: image
  - name: collection
    dtype: string
  - name: quality_assessment
    struct:
    - name: compositeStructure
      dtype: int64
    - name: objectConsistency
      dtype: int64
    - name: imageQuality
      dtype: int64
  - name: description
    struct:
    - name: item
      dtype: string
    - name: description_0
      dtype: string
    - name: description_1
      dtype: string
    - name: category
      dtype: string
    - name: description_valid
      dtype: bool
  splits:
  - name: train
    num_bytes: 15936399912.472
    num_examples: 206841
  download_size: 10553550156
  dataset_size: 15936399912.472
---

<img src='https://github.com/Yuanshi9815/Subjects200K/raw/main/assets/data.jpg' width='100%' />

<a href="https://github.com/Yuanshi9815/OminiControl"><img src="https://img.shields.io/badge/GitHub-OminiControl-blue.svg?logo=github&" alt="GitHub"></a>

Subjects200K is a large-scale dataset containing 200,000 paired images, introduced as part of the [OminiControl](https://github.com/Yuanshi9815/OminiControl) project. Each image pair maintains subject consistency while presenting variations in scene context.


### Quick Start
- Load dataset
  ```python
  from datasets import load_dataset

  # Load dataset
  dataset = load_dataset('Yuanshi/Subjects200K')
  ```

- Filter high-quality pairs from `collection_2`
  ```python
  def filter_func(item):
      if item.get("collection") != "collection_2":
          return False
      if not item.get("quality_assessment"):
          return False
      return all(
          item["quality_assessment"].get(key, 0) >= 5
          for key in ["compositeStructure", "objectConsistency", "imageQuality"]
      )
  
  collection_2_valid = dataset["train"].filter(
      filter_func,
      num_proc=16,
      cache_file_name="./cache/dataset/collection_2_valid.arrow", # Optional
  )
  ```


### Collections
**Collection1 (`collection_1`)**
-  512 x 512 resolution, with 16-pixel padding.
-  Total 18,396 image pairs, with 8,200 pairs having high-quality ratings.

**Collection2 (`collection_2`)**
-  512 x 512 resolution, with 16-pixel padding.
-  Total 187,840 image pairs, with 111,767 pairs having high-quality ratings.

**Collection3 (`collection_3`)** [link](https://huggingface.co/datasets/Yuanshi/Subjects200K_collection3)
-  1024 x 1024 resolution.

>  The description formats may vary across different collections.

### Data Format
  | Key name             | Type    | Description                                                                                                                                                                                                |
  | -------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `image`              | `image` | A composite image containing a pair of images with 16-pixel padding.                                                                                                                                       |
  | `collection`         | `str`   | The name or identifier of the source collection.                                                                                                                                                           |
  | `quality_assessment` | `dict`  | Quality scores evaluated by the ChatGPT-4o model. Each quality dimension is rated on a scale of 0-5. Recommended for filtering out low-quality image pairs which do not have all dimensions rated above 5. |
  | `description`        | `dict`  | Detailed textual description of the image pair contents.                                                                                                                                                   |



## Contributing
We welcome contributions! Please feel free to submit a Pull Request or open an Issue.

## Citation
```
@article{
  tan2024omini,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang},
  journal={arXiv preprint arXiv:2411.15098},
  year={2024}
}
```



This repo contains the dataset used in [OminiControl: Minimal and Universal Control for Diffusion Transformer](https://huggingface.co/papers/2411.15098).

---

# Subjects200K_collection3

---
dataset_info:
  features:
  - name: quality_assessment
    struct:
    - name: compositeStructure
      dtype: int64
    - name: objectConsistency
      dtype: int64
    - name: imageQuality
      dtype: int64
  - name: description
    struct:
    - name: item
      dtype: string
    - name: description_0
      dtype: string
    - name: description_1
      dtype: string
    - name: category
      dtype: string
    - name: description_valid
      dtype: bool
  - name: raw_json
    dtype: string
  - name: image
    dtype: image
  - name: collection
    dtype: string
  splits:
  - name: train
    num_bytes: 59403928706.041
    num_examples: 49159
  download_size: 59353522419
  dataset_size: 59403928706.041
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
<img src='https://github.com/Yuanshi9815/Subjects200K/raw/main/assets/data.jpg' width='100%' />

<a href="https://github.com/Yuanshi9815/OminiControl"><img src="https://img.shields.io/badge/GitHub-OminiControl-blue.svg?logo=github&" alt="GitHub"></a>

Subjects200K is a large-scale dataset containing 200,000 paired images, introduced as part of the [OminiControl](https://github.com/Yuanshi9815/OminiControl) project. Each image pair maintains subject consistency while presenting variations in scene context.


### Quick Start
- Load dataset
  ```python
  from datasets import load_dataset

  # Load dataset
  dataset = load_dataset('Yuanshi/Subjects200K')
  ```

- Filter high-quality pairs from `collection_2`
  ```python
  def filter_func(item):
      if item.get("collection") != "collection_2":
          return False
      if not item.get("quality_assessment"):
          return False
      return all(
          item["quality_assessment"].get(key, 0) >= 5
          for key in ["compositeStructure", "objectConsistency", "imageQuality"]
      )
  
  collection_2_valid = dataset["train"].filter(
      filter_func,
      num_proc=16,
      cache_file_name="./cache/dataset/collection_2_valid.arrow", # Optional
  )
  ```


### Collections
**Collection1 (`collection_1`)** [link](https://huggingface.co/datasets/Yuanshi/Subjects200K)
-  512 x 512 resolution, with 16-pixel padding.
-  Total 18,396 image pairs, with 8,200 pairs having high-quality ratings.

**Collection2 (`collection_2`)** [link](https://huggingface.co/datasets/Yuanshi/Subjects200K)
-  512 x 512 resolution, with 16-pixel padding.
-  Total 187,840 image pairs, with 111,767 pairs having high-quality ratings.

**Collection3 (`collection_3`)** 
-  1024 x 1024 resolution.

>  The description formats may vary across different collections.

### Data Format
  | Key name             | Type    | Description                                                                                                                                                                                                |
  | -------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `image`              | `image` | A composite image containing a pair of images with 16-pixel padding.                                                                                                                                       |
  | `collection`         | `str`   | The name or identifier of the source collection.                                                                                                                                                           |
  | `quality_assessment` | `dict`  | Quality scores evaluated by the ChatGPT-4o model. Each quality dimension is rated on a scale of 0-5. Recommended for filtering out low-quality image pairs which do not have all dimensions rated above 5. |
  | `description`        | `dict`  | Detailed textual description of the image pair contents.                                                                                                                                                   |



## Contributing
We welcome contributions! Please feel free to submit a Pull Request or open an Issue.

## Citation
```
@article{
  tan2024omini,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang},
  journal={arXiv preprint arXiv:2411.15098},
  year={2024}
}
```



This repo contains the dataset used in [OminiControl: Minimal and Universal Control for Diffusion Transformer](https://huggingface.co/papers/2411.15098).

