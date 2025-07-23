import os
import json
from trig.utils import load_config, base64_to_image
from trig.config import DIM_DICT_WITHOUT_M_E
from trig.models import import_model
from pathlib import Path
import shutil
import requests
import multiprocessing
from tqdm import tqdm
from datasets import load_dataset

project_root = Path(__file__).resolve().parents[2]

# export PYTHONPATH=/home/muzammal/Projects/TRIG:$PYTHONPATH

class Generator:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)
        print("-" * 50)
        print("Experiment name:", self.config["name"])
        print("Task:", self.config["task"])
        print("Models:", self.config["generation"]["models"])
        self.prompts_data = self.load_prompts()
        if "description_path" in self.config:
            self.description_data = self.load_descriptions(self.config["description_path"])
        print("-" * 50)

    def instantiate_models(self):
        models = {}
        for model_name in self.config["generation"]["models"]:
            model_class = import_model(model_name)
            models[model_name] = model_class()

        print(f"Models loaded: {models.keys()}")
        return models

    def load_prompts(self):
        if "prompt_path" in self.config:
            # Not recommended, use datasets instead
            with open(self.config["prompt_path"], 'r') as file:
                prompts_data = json.load(file)

            # check details
            total_len = len(prompts_data)
            uni_len = sum(
                1
                for prompt in prompts_data
                if "parent_dataset" in prompt
                and len(prompt["parent_dataset"]) == 2
                and prompt["parent_dataset"][0].startswith("<")
                and prompt["parent_dataset"][0].endswith(">")
                and prompt["parent_dataset"][1] == "Origin"
            )
            print(total_len, uni_len)

            return prompts_data
        else:
            split_dir = {
                "t2i": "text_to_image",
                "p2p": "image_editing",
                "s2p": "subject_driven",
            }
            print("Loading dataset from Hugging Face, task: ", split_dir[self.config["task"]])
            prompts_data = load_dataset("TRIG-bench/TRIG", split=split_dir[self.config["task"]])
            print("Dataset loaded")
            return prompts_data
    
    

    def load_descriptions(self, description_file):
        # deprecated
        with open(description_file, 'r') as file:
            description_data = json.load(file)
        return description_data

    def generate_batch_models(self, start_idx=None, end_idx=None):
        # FIXME: multiprocessing not working
        # with multiprocessing.Pool() as pool:
        #     pool.map(self.generate_single_model, self.config["generation"]["models"])
        for model_name in self.config["generation"]["models"]:
            self.generate_single_model(model_name, start_idx, end_idx)

    def save_image(self, image, output_path, filename):
        file_path = os.path.join(output_path, f"{filename}.png")

        try:
            if isinstance(image, str):
                response = requests.get(image)
                response.raise_for_status()
                with open(file_path, "wb") as file:
                    file.write(response.content)
            else: 
                image.save(file_path)
        except Exception as e:
            print(f"Failed to save image: {filename}, Error: {e}")

    def generate_single_model(self, model_name, start_idx=None, end_idx=None):
        model_class = import_model(model_name)
        model = model_class()
        task  = self.config["task"]
        
        output_path = os.path.join(project_root, 'data/output/', task, model_name)
        print(f"Output path: {output_path}")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_list = os.listdir(output_path)
        file_list = [os.path.splitext(f)[0] for f in file_list]

        if start_idx is not None and end_idx is not None:
            if end_idx > len(self.prompts_data):
                end_idx = len(self.prompts_data)
            self.prompts_data = self.prompts_data[start_idx:end_idx]

        for prompt_data in tqdm(self.prompts_data):
            # TODO: Align HF dataset format
            if prompt_data["data_id"] in file_list:
                continue

            # if  "IQ-R_R-T" in prompt_data["data_id"] and model_name=='dalle3':
            #     continue

            # if prompt_data["parent_dataset"][0].startswith("<") and prompt_data["parent_dataset"][0].endswith(
            #         ">") and prompt_data["parent_dataset"][1] == "Origin":
            #     parent_dim1 = prompt_data["parent_dataset"][0].split(", ")[0]
            #     parent_dim2 = prompt_data["parent_dataset"][0].split(", ")[1]
            #     parent_dim = f"{parent_dim1[1:]}_{parent_dim2[:-1]}"
            #     idx = prompt_data['data_id'].split('_')[-1]

            #     if os.path.exists(
            #             os.path.join(output_path, f'{parent_dim}_{idx}.png')):
            #         shutil.copy(os.path.join(output_path, f'{parent_dim}_{idx}.png'),
            #                     os.path.join(output_path, f"{prompt_data['data_id']}.png"))
            #     else:
            #         print(f"Parent image not found: {parent_dim}_{idx}.png")
            #         continue

            
            if "image_path" in self.config:
                # if use json (not recommended), image is a path to the image
                image = os.path.join(self.config["image_path"], prompt_data["img_id"])
            else:
                # from HF dataset, image is a PIL Image object
                image = prompt_data["image"].convert("RGB")

            prompt = prompt_data["prompt"]
            if "dimensions" in prompt_data:
                dimensions = prompt_data["dimensions"]
            item = prompt_data["item"] if "item" in prompt_data else None

            #  deprecated
            # if model_name == "flowedit":
            #     src_prompt = description_data[prompt_data["img_id"]]

            task_mapping = {
                "t2i": lambda: model.generate(prompt),
                "p2p": lambda: model.generate_p2p(prompt, image),
                "s2p": lambda: model.generate_s2p(prompt, item, image),
                "t2i_dtm": lambda: model.generate(prompt, dimensions),
                "p2p_dtm": lambda: model.generate_p2p(prompt, image, dimensions),
                "s2p_dtm": lambda: model.generate_s2p(prompt, item, image, dimensions),
            }
            image = task_mapping[task]()

            self.save_image(image, output_path, prompt_data["data_id"])


if __name__ == "__main__":
    generator = Generator(config_path=r"/home/zsc/TRIG/config/gen.yaml")
    generator.generate_batch_models()
