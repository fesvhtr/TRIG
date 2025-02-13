from collections import defaultdict
import yaml
import os
import importlib
from pathlib import Path
import json
from natsort import natsorted
from trig.metrics import import_metric

project_root = Path(__file__).resolve().parents[2]


class Evaluator:
    def __init__(self, config_path="config/default.yaml"):
        self.prompt_dic = None
        self.config = self.load_config(config_path)
        self.metrics = self.instantiate_metrics()
        self.prompt_dic = self.load_prompts()

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_prompts(self):
        prompt_path = self.config["evaluation"]["prompt_path"]
        with open(prompt_path, "r") as f:
            data = json.load(f)
        prompt_dic = {}
        for i in data:
            prompt_dic[i['data_id']] = {
                'prompt': i['prompt'],
                'sub_prompt': i['dimension_prompt'],
                'img_id': i['img_id']  # ground truth image
            }
        return prompt_dic

    def instantiate_metrics(self):
        """
        Dynamically instantiate multiple metrics for each dimension.
        """
        dim_metrics = {}
        dimensions = self.config["dimensions"]

        for dim, config in dimensions.items():
            metrics = config.get("metrics", [])
            dim_metrics = []

            for metric_name in metrics:
                metric_class = import_metric(metric_name)
                dim_metrics.append(metric_class())

            dim_metrics[dim] = dim_metrics  # metrics list for each dimension

        print(f"Metrics loaded for dimensions: {dim_metrics.keys()}")
        return dim_metrics

    def parse_dimensions(self, filename):
        base_name = os.path.splitext(filename)[0]
        dimensions = base_name.split("_")[:-1]
        return dimensions

    def save_results(self, results):
        output_path = os.path.join(project_root, self.config['evaluation']['result_dir'], f"{self.config['name']}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results Saved:'{output_path}'")

    def group_images_by_combination(self, image_dir):
        grouped = defaultdict(lambda: {"data_ids": [], "image_paths": [], "prompts": []})
        images = natsorted([f for f in os.listdir(os.path.join(project_root, image_dir))])
        print(f"Found {len(images)} images in '{image_dir}'")

        for filename in images:
            combination = "_".join(self.parse_dimensions(filename))
            image_path = os.path.join(image_dir, filename)
            data_id = os.path.splitext(filename)[0]
            prompt = self.prompt_dic[data_id]
            grouped[combination]["data_ids"].append(data_id)
            grouped[combination]["image_paths"].append(image_path)
            grouped[combination]["prompts"].append(prompt)

        return grouped

    def evaluate_dim_pair(self, combination, data):
        combined_scores = defaultdict(list)
        for dim in combination.split("_"):
            if dim not in self.metrics:
                raise ValueError(f"Dimension '{dim}' not found in metrics!")
            metric = self.metrics[dim]
            scores = metric.compute_batch(data["data_ids"], data["image_paths"], data["prompts"])
            for key, value in scores.items():
                combined_scores[key].append(value)
        return combined_scores

    def evaluate_all(self):
        image_dir = self.config["evaluation"]["image_dir"]
        grouped = self.group_images_by_combination(image_dir)
        # print(grouped)

        final_results = {}
        for combination, prompts in grouped.items():
            print(f"Evaluating combination {combination} with {len(prompts['image_paths'])} images...")
            combined_scores = self.evaluate_dim_pair(combination, prompts)
            # print(combined_scores)
            final_results[combination] = combined_scores
        self.save_results(final_results)
        print("Evaluation complete!")


if __name__ == "__main__":
    evaluator = Evaluator(config_path=r"/home/muzammal/Projects/TRIG/config/eval.yaml")

    evaluator.evaluate_all()
