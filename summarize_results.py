import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_all_results(outputs_root: str) -> List[Dict]:
    records: List[Dict] = []
    pattern = os.path.join(outputs_root, "*", "results.json")
    for path in glob.glob(pattern):
        with open(path, "r") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        dataset_name = cfg.get("dataset", {}).get("name", "unknown")
        model_name = cfg.get("model", {}).get("name", "unknown")
        test_metrics = data.get("test_metrics", {})
        records.append(
            {
                "experiment_dir": os.path.dirname(path),
                "dataset": dataset_name,
                "architecture": model_name,
                "accuracy": test_metrics.get("accuracy", None),
                "f1": test_metrics.get("f1", None),
                "training_time_sec": data.get("training_time_sec", None),
                "param_count": data.get("param_count", None),
            }
        )
    return records


def print_markdown_table(records: List[Dict]) -> None:
    print("| Dataset | Architecture | Accuracy | F1 | Training Time (s) | Params | Notes |")
    print("|---------|-------------|----------|----|--------------------|--------|-------|")
    for r in records:
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else ""
        f1 = f"{r['f1']:.4f}" if r["f1"] is not None else ""
        t = f"{r['training_time_sec']:.1f}" if r["training_time_sec"] is not None else ""
        params = f"{r['param_count']:,}" if r["param_count"] is not None else ""
        print(
            f"| {r['dataset']} | {r['architecture']} | {acc} | {f1} | {t} | {params} | |"
        )


def plot_param_vs_performance(records: List[Dict], outputs_root: str) -> None:
    # Group by dataset
    by_dataset: Dict[str, List[Dict]] = {}
    for r in records:
        by_dataset.setdefault(r["dataset"], []).append(r)

    for dataset, recs in by_dataset.items():
        archs = [r["architecture"] for r in recs]
        params = [r["param_count"] or 0 for r in recs]
        accs = [r["accuracy"] or 0.0 for r in recs]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        x = range(len(archs))
        ax1.bar(x, params, color="tab:blue", alpha=0.6, label="Params")
        ax2.plot(x, accs, color="tab:red", marker="o", label="Accuracy")

        ax1.set_xlabel("Architecture")
        ax1.set_ylabel("Parameter Count", color="tab:blue")
        ax2.set_ylabel("Accuracy", color="tab:red")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(archs)
        plt.title(f"Parameter Count vs Accuracy — {dataset}")

        fig.tight_layout()
        out_path = os.path.join(outputs_root, f"{dataset}_param_vs_accuracy.png")
        plt.savefig(out_path)
        plt.close(fig)


def main() -> None:
    outputs_root = "outputs"
    records = _load_all_results(outputs_root)
    if not records:
        print("No results.json files found under outputs/.")
        return

    print("\nMarkdown table for report:\n")
    print_markdown_table(records)

    print("\nGenerating parameter vs performance plots...")
    plot_param_vs_performance(records, outputs_root)
    print("Done. Check PNG files in the outputs/ directory.")


if __name__ == "__main__":
    main()

