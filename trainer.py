from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class History:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    is_train: bool,
) -> Tuple[float, float]:
    if is_train:
        model.train()
    else:
        model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    running_loss = 0.0
    n_samples = 0

    for inputs, targets in tqdm(loader, disable=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()  # type: ignore[arg-type]

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()  # type: ignore[union-attr]

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    epoch_loss = running_loss / max(n_samples, 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)

    if num_classes == 2:
        f1 = f1_score(all_targets, all_preds)
    else:
        f1 = f1_score(all_targets, all_preds, average="macro")

    return {"accuracy": acc, "f1": f1}


def _maybe_plot_curves(history: History, output_dir: str) -> None:
    epochs = list(range(1, len(history.train_loss) + 1))

    # Loss curves
    plt.figure()
    plt.plot(epochs, history.train_loss, label="train_loss")
    plt.plot(epochs, history.val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, history.train_acc, label="train_acc")
    plt.plot(epochs, history.val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curves.png"))
    plt.close()


def _maybe_save_conv_weights(model: nn.Module, output_dir: str) -> None:
    """Save a visualization of the first Conv2d filters if present."""
    first_conv2d = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv2d = m
            break

    if first_conv2d is None:
        return

    weight = first_conv2d.weight.detach().cpu().clone()  # (out_c, in_c, k, k)
    out_channels = weight.shape[0]
    num_show = min(out_channels, 16)

    # Normalize each filter to [0, 1]
    w = weight[:num_show]
    w_min = w.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    w_max = w.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    w = (w - w_min) / (w_max - w_min + 1e-8)

    cols = 4
    rows = int((num_show + cols - 1) / cols)
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_show):
        plt.subplot(rows, cols, i + 1)
        # If in_channels == 3, show RGB; otherwise, average over channels
        img = w[i]
        if img.shape[0] == 3:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.mean(dim=0).numpy()
        plt.imshow(img_np, cmap=None if img.shape[0] == 3 else "viridis")
        plt.axis("off")
    plt.suptitle("First Conv2d Filters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conv_filters.png"))
    plt.close()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg,
    num_classes: int,
    output_dir: str,
) -> Tuple[nn.Module, History, Dict[str, float]]:
    os.makedirs(output_dir, exist_ok=True)

    training_cfg = cfg["training"]
    logging_cfg = cfg.get("logging", {})

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if training_cfg["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_cfg["lr"],
            weight_decay=training_cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_cfg["lr"],
            momentum=0.9,
            weight_decay=training_cfg["weight_decay"],
        )

    num_epochs = training_cfg["num_epochs"]
    early_cfg = training_cfg["early_stopping"]
    use_early = early_cfg.get("enabled", False)
    patience = early_cfg.get("patience", 5)
    min_delta = early_cfg.get("min_delta", 0.0)

    history = History([], [], [], [])
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    train_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, criterion, optimizer, is_train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, device, criterion, optimizer=None, is_train=False
        )

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if use_early and epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    train_end_time = time.time()
    training_time_sec = train_end_time - train_start_time
    avg_epoch_time_sec = training_time_sec / max(len(history.train_loss), 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_start_time = time.time()
    test_metrics = evaluate_metrics(model, test_loader, device, num_classes)
    eval_end_time = time.time()
    test_eval_time_sec = eval_end_time - eval_start_time

    print("Test metrics:", test_metrics)
    print(f"Total training time (s): {training_time_sec:.2f}")

    # Compute parameter count
    param_count = sum(p.numel() for p in model.parameters())

    # Optionally save best model state
    if logging_cfg.get("save_best_model", False):
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # Optionally generate per-experiment plots and conv filter visualizations
    if logging_cfg.get("save_plots", False):
        _maybe_plot_curves(history, output_dir)
        _maybe_save_conv_weights(model, output_dir)

    # Save metrics and history
    results = {
        "history": asdict(history),
        "test_metrics": test_metrics,
        "config": cfg,
        "training_time_sec": training_time_sec,
        "avg_epoch_time_sec": avg_epoch_time_sec,
        "test_eval_time_sec": test_eval_time_sec,
        "param_count": param_count,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, history, test_metrics


