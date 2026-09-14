from src.model.lightning import get_trainer
from src.config import Config
import argparse
import yaml
from pathlib import Path
import os
import shutil
import src.dataset as datasets
import src.model.flow_matching_model as flow_matching

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with config file")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configurations/gaussian_prior_rayleigh_drw.yaml",
        help="Path to YAML/JSON configuration file",
    )
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return Config(**yaml_dict)

def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path("outputs")/config.wandb.name
    os.makedirs(str(output_dir), exist_ok=True)
    shutil.copy(args.config, str(output_dir/Path("config.yaml")))

    data_module = getattr(datasets, config.data.data_class)(config.data)
    wrapper = getattr(flow_matching, config.lightning_wrapper)(config)
    trainer = get_trainer(config)
    trainer.fit(wrapper, data_module)

if __name__ == "__main__":
    main()
