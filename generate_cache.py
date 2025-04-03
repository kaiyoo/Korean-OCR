import os
import sys
import yaml
import paddle
from tqdm import tqdm
from ppocr.data.simple_dataset import CachedSimpleDataSet
from ppocr.utils.logging import get_logger

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_cache(config_path, mode):
    logger = get_logger()
    config = load_config(config_path)
    logger.info(f"Loading config from {config_path}")

    dataset = CachedSimpleDataSet(config, mode, logger)
    logger.info(f"Start generating cache for {mode} dataset, total {len(dataset)} samples")

    pbar = tqdm(total=len(dataset), desc=f"Generating {mode} cache")
    for idx in range(len(dataset)):
        _ = dataset[idx]
        pbar.update(1)
    pbar.close()

    logger.info(f"✅ Cache generation completed for {mode} dataset.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config.yml")
    args = parser.parse_args()

    # Train, Eval 캐시 생성
    generate_cache(args.config, "Train")
    generate_cache(args.config, "Eval")
