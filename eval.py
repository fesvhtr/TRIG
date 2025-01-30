from trig.evaluation.evaluator import Evaluator
from trig.evaluation.generator import Generator
import yaml

if __name__ == "__main__":

    # 读取 YAML 文件
    with open(r"H:\ProjectsPro\TRIG\config\demo.yaml", "r") as file:
        config = yaml.safe_load(file)

    if "generation" in config:
        generator = Generator(config_path=r"H:\ProjectsPro\TRIM\config\demo.yaml")

    if "evaluation" in config:
        evaluator = Evaluator(config_path=r"H:\ProjectsPro\TRIM\config\demo.yaml")
        final_results = evaluator.evaluate_all()
        print(final_results)

    if "relation" in config:
        pass