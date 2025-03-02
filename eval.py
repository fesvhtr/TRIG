import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse


if __name__ == "__main__":
  

    parser = argparse.ArgumentParser(description="Run TRIG program with specified configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/muzammal/Projects/TRIG/config/flux.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        help="Index to start from."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        help="Index to end at."
    )
    args = parser.parse_args()
    
    config_path = args.config
    print(f"Using config file: {config_path}")
    # read yaml file in config folder
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # step 1: generate images
    if "generation" in config:
        from trig.evaluation.generator import Generator
        generator = Generator(config_path=config_path)
        generator.generate_batch_models(start_idx=args.start_idx, end_idx=args.end_idx)

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