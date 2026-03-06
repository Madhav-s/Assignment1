import copy
import os

import torch

from datasets import build_dataloaders
from models import create_model
from run_experiment import load_config, set_seed
from trainer import train_model


EXPERIMENTS = [
    ("adult", "mlp", "adult_mlp"),
    ("adult", "cnn", "adult_cnn"),
    ("adult", "attention", "adult_attention"),
    ("cifar10", "mlp", "cifar10_mlp"),
    ("cifar10", "cnn", "cifar10_cnn"),
    ("cifar10", "attention", "cifar10_attention"),
    ("pcam", "mlp", "pcam_mlp"),
    ("pcam", "cnn", "pcam_cnn"),
    ("pcam", "attention", "pcam_attention"),
]


def main() -> None:
    base_cfg = load_config("config.yaml")
    device_str = base_cfg["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    for dataset_name, model_name, exp_name in EXPERIMENTS:
        cfg = copy.deepcopy(base_cfg)
        cfg["dataset"]["name"] = dataset_name
        cfg["model"]["name"] = model_name
        cfg["experiment_name"] = exp_name

        set_seed(cfg.get("seed", 42))

        train_loader, val_loader, test_loader, info = build_dataloaders(
            cfg, dataset_name
        )

        model = create_model(
            model_name=model_name,
            input_shape=info.input_shape,
            num_classes=info.num_classes,
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
            use_batchnorm=cfg["model"]["use_batchnorm"],
            dataset_name=dataset_name,
        )

        output_root = cfg["logging"]["output_dir"]
        output_dir = os.path.join(output_root, exp_name)

        print(
            f"\n=== Running experiment {exp_name} "
            f"(dataset={dataset_name}, model={model_name}) ==="
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

