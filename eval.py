from trig.evaluation.evaluator import Evaluator
from trig.evaluation.generator import Generator
from trig.evaluation.relationator import Relationator
import yaml

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":

    config_path = '/Users/binzhu/Desktop/Paper/GGBench/TRIG/config/relation.yaml'
    # read yaml file in config folder
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # step 1: generate images
    if "generation" in config:
        generator = Generator(config_path=config_path)
        generator.generate_batch_models()

    # step 2: evaluate images
    if "evaluation" in config:
        evaluator = Evaluator(config_path=config_path)
        evaluator.evaluate_all()

    # step 3: build relation
    if "relation" in config:
        relation = Relationator(config_path=config_path)
        relation.build_relation()