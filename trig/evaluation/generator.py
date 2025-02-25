import os
import json
from trig.utils import load_config, base64_to_image
from trig.config import DIM_DICT_WITHOUT_M_E
from trig.models import import_model
from pathlib import Path
import shutil
import multiprocessing
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]

# export PYTHONPATH=/home/muzammal/Projects/TRIG:$PYTHONPATH

class Generator:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)
        print("-" * 50)
        print("Experiment name:", self.config["name"])
        print("Task:", self.config["task"])
        print("Models:", self.config["generation"]["models"])
        self.prompts_data = self.load_prompts(self.config["prompt_path"])
        print("-" * 50)

    def instantiate_models(self):
        models = {}
        for model_name in self.config["generation"]["models"]:
            model_class = import_model(model_name)
            models[model_name] = model_class()

        print(f"Models loaded: {models.keys()}")
        return models

    def load_prompts(self, prompt_file):
        with open(prompt_file, 'r') as file:
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

    def generate_batch_models(self):
        # FIXME: multiprocessing not working
        # with multiprocessing.Pool() as pool:
        #     pool.map(self.generate_single_model, self.config["generation"]["models"])
        for model_name in self.config["generation"]["models"]:
            self.generate_single_model(model_name)

    def generate_single_model(self, model_name):
        model_class = import_model(model_name)
        model = model_class()
        task  = self.config["task"]
        output_path = os.path.join(project_root, 'data/output', task, model_name)
        print(f"Output path: {output_path}")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_list = os.listdir(output_path)
        file_list = [os.path.splitext(f)[0] for f in file_list]

        for prompt_data in tqdm(self.prompts_data[6000:8000]):
            if prompt_data["data_id"] in file_list:
                continue

            if prompt_data["parent_dataset"][0].startswith("<") and prompt_data["parent_dataset"][0].endswith(
                    ">") and prompt_data["parent_dataset"][1] == "Origin":
                parent_dim1 = prompt_data["parent_dataset"][0].split(", ")[0]
                parent_dim2 = prompt_data["parent_dataset"][0].split(", ")[1]
                parent_dim = f"{parent_dim1[1:]}_{parent_dim2[:-1]}"
                idx = prompt_data['data_id'].split('_')[-1]

                if os.path.exists(
                        os.path.join(output_path, f'{parent_dim}_{idx}.png')):
                    shutil.copy(os.path.join(output_path, f'{parent_dim}_{idx}.png'),
                                os.path.join(output_path, f"{prompt_data['data_id']}.png"))
                else:
                    print(f"Parent image not found: {parent_dim}_{idx}.png")
                    continue

            else:
                prompt = prompt_data["prompt"]
                if task == "t2i":
                    image = model.generate(prompt)
                elif task == "p2p":
                    image_path = os.path.join(self.config["image_path"], prompt_data["img_id"])
                    image = model.generate_p2p(prompt, image_path)
                elif task == "s2p":
                    image = model.generate_s2p()
                else:
                    raise ValueError(f"Task {task} not supported")

                image.save(os.path.join(output_path, f"{prompt_data['data_id']}.png"))


if __name__ == "__main__":
    generator = Generator(config_path=r"/home/zsc/TRIG/config/gen.yaml")
    generator.generate_batch_models()
