from trig.evaluation.evaluator import Evaluator
from trig.evaluation.generator import Generator
import yaml

if __name__ == "__main__":

    config_path = r"H:\ProjectsPro\TRIM\config\demo.yaml"
    # read yaml file in config folder
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # step 1: generate images
    if "generation" in config:
        generator = Generator(config_path=config_path)
        generator.generate_all()

    # step 2: evaluate images
    if "evaluation" in config:
        evaluator = Evaluator(config_path=config_path)
        evaluator.evaluate_all()

    # step 3: build relation
    if "relation" in config:
        pass