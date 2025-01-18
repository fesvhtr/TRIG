import yaml
import importlib

class Evaluator:
    def __init__(self, config_path="config/metrics.yaml"):
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_metric_class(self, module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            metric_class = getattr(module, class_name)
            return metric_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error loading metric class {class_name} from {module_name}: {e}")

    def evaluate(self, prompts, images):

        results = {}
        dimensions = self.config.get("dimensions", {})

        for dim, config in dimensions.items():
            module_name = config["module"]
            class_name = config["class"]
            print(f"Evaluating {dim} using {class_name}...")

            # 动态加载 Metric 类
            metric_class = self.load_metric_class(module_name, class_name)

            # 实例化 Metric 类
            metric_instance = metric_class(**config.get("params", {}))

            # 调用 Metric 的 evaluate 方法
            scores = metric_instance.evaluate(prompts, images)
            results[dim] = {
                "name": config["name"],
                "metric": class_name,
                "scores": scores,
                "average_score": sum(scores) / len(scores) if scores else 0
            }

        return results
