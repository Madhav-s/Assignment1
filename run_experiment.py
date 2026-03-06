import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml

from datasets import build_dataloaders
from models import create_model
from trainer import train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DL assignment experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model"]["name"]
    hidden_dims = cfg["model"]["hidden_dims"]
    dropout = cfg["model"]["dropout"]
    use_batchnorm = cfg["model"]["use_batchnorm"]

    device_str = cfg["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    set_seed(cfg.get("seed", 42))

    train_loader, val_loader, test_loader, info = build_dataloaders(cfg, dataset_name)

    model = create_model(
        model_name=model_name,
        input_shape=info.input_shape,
        num_classes=info.num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        dataset_name=dataset_name,
    )

    exp_name = cfg.get(
        "experiment_name", f"{dataset_name}_{model_name}"
    )
    output_root = cfg["logging"]["output_dir"]
    output_dir = os.path.join(output_root, exp_name)

    print(
        f"Running experiment: {exp_name}\n"
        f"Dataset: {dataset_name}, Model: {model_name}, Device: {device}"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        cfg=cfg,
        num_classes=info.num_classes,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

