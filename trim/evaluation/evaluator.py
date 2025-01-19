from collections import defaultdict
import yaml
import os
import importlib
class Evaluator:
    def __init__(self, config_path="config/default.yaml"):
        self.prompt_dic = None
        self.config = self.load_config(config_path)
        self.metrics = self.instantiate_metrics()

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_prompts(self):
        prompt_path = 'data/dataset/prompts/trim_{}.json'.format(self.config["evaluation"]["task"])
        with open(prompt_path, "r") as f:
            data = json.load(f)
        prompt_dic = {}
        for i in data:
            prompt_dic[i['data_id']] = {
                'prompt': i['prompt'],
                'sub_prompt': i['dimension_prompt'],
                'img_id': i['img_id']  # ground truth image
            }
        self.prompt_dic = prompt_dic

    def instantiate_metrics(self):
        metrics = {}
        dimensions = self.config["dimensions"]
        for dim, config in dimensions.items():
            module_name = config["module"]
            class_name = config["class"]
            params = config.get("params", {})

            # 动态加载模块和类
            module = importlib.import_module(module_name)
            metric_class = getattr(module, class_name)
            metrics[dim] = metric_class(**params)

        return metrics

    def parse_dimensions(self, filename):
        base_name = os.path.splitext(filename)[0]
        dimensions = base_name.split("_")[:-1]
        return dimensions

    def save_results(self, results):
        output_dir = os.path.join(self.config["evaluation"]["output_dir"], self.config["evaluation"]["name"])
        os.makedirs(output_dir, exist_ok=True)
        for result in results:
            image_name = result["image"]
            output_path = os.path.join(output_dir, image_name + ".json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

    def group_images_by_combination(self, image_dir):
        grouped = defaultdict(lambda: {"image_paths": [], "prompts": []})
        images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

        for filename in images:
            combination = "_".join(self.parse_filename(filename))
            image_path = os.path.join(image_dir, filename)

            # 从文件名提取 data_id，并从 prompt_dic 中获取对应的 prompt
            data_id = os.path.splitext(filename)[0]  # 去掉扩展名作为 data_id
            if data_id not in self.prompt_dic:
                raise KeyError(f"Prompt for data_id '{data_id}' not found in prompt_dic!")

            prompt = self.prompt_dic[data_id]
            grouped[combination]["image_paths"].append(image_path)
            grouped[combination]["prompts"].append(prompt)

        return grouped

    def evaluate_batch(self, combination, image_paths, prompts):

        results = {}
        dimensions = combination.split("_")

        for dim in dimensions:
            if dim not in self.metrics:
                raise ValueError(f"Dimension '{dim}' not found in metrics!")
            metric = self.metrics[dim]

            scores = metric.compute_batch(prompts, image_paths)
            results[dim] = scores

        return results

    def evaluate_all(self):
        image_dir = self.config["model"]["output_dir"]
        grouped_images = self.group_images_by_combination(image_dir)
        final_results = []

        for combination, data in grouped_images.items():
            print(f"Evaluating combination {combination} with {len(data['image_paths'])} images...")

            batch_results = self.evaluate_batch(
                combination,
                data["image_paths"],
                data["prompts"]
            )

            for img_path, prompt in zip(data["image_paths"], data["prompts"]):
                image_name = os.path.basename(img_path)
                image_results = {dim: batch_results[dim][i] for i, dim in enumerate(combination.split("_"))}
                final_results.append({
                    "image": image_name,
                    "prompt": prompt,
                    "results": image_results
                })

        return final_results


# 示例主逻辑
if __name__ == "__main__":

    evaluator = Evaluator(config_path=r"H:\ProjectsPro\TRIM\config\test.yaml")

    final_results = evaluator.evaluate_all()

    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("Evaluation completed. Results saved to 'evaluation_results.json'.")
