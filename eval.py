import yaml
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Run evaluation with config file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config yaml file')
    args = parser.parse_args()

    # 使用命令行参数中的配置文件路径
    config_path = args.config
    # read yaml file in config folder
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # step 1: generate images
    if "generation" in config:
        from trig.evaluation.generator import Generator
        generator = Generator(config_path=config_path)
        generator.generate_batch_models()

    # step 2: evaluate images
    if "evaluation" in config:
        from trig.evaluation.evaluator import Evaluator
        evaluator = Evaluator(config_path=config_path)
        evaluator.evaluate_all()

    # step 3: build relation
    if "relation" in config:
        from trig.evaluation.relationator import Relationator
        relation = Relationator(config_path=config_path)
        relation.build_relation()