### Deep Learning Assignment 1 — Datasets × Architectures Benchmark

This repository contains a complete PyTorch implementation for the **3 datasets × 3 architectures** benchmark described in your assignment:

- **Dataset A**: UCI Adult Income (tabular, binary classification)
- **Dataset B**: CIFAR-10 (natural images, 10 classes)\*
- **Dataset C**: PatchCamelyon (PCam, histopathology images, binary classification)

\*The PDF mentions CIFAR-100 with 10 classes; here we **assume CIFAR-10** (the standard 10‑class benchmark). You can switch to CIFAR‑100 with minor changes if required.

The project is designed to be:

- **Config‑driven** (datasets, architectures, batch size, learning rate, etc. are all configurable)
- **Reproducible** (single entrypoint script, fixed seeds in config)
- **Modular** (separate modules for datasets, models, and training)

---

### 1. Setup

#### 1.1. Create and activate environment

Use any environment manager (conda, venv, etc.). Example with `venv`:

```bash
cd dl_assignment1
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

#### 1.2. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure your PyTorch / TorchVision versions match your CUDA / CPU setup (follow the official PyTorch install instructions if needed).

---

### 2. Datasets

#### 2.1. UCI Adult Income (Tabular)

- Download the **Adult** dataset from UCI or OpenML and save it as:
  - `data/adult/adult.csv`
- Ensure the target column is named `income` with values containing `>50K` and `<=50K` (the loader treats `>50K` as the positive class).

The loader:

- One‑hot encodes categorical features
- Standard‑scales numeric features
- Builds **train / validation / test** splits using the ratios in `config.yaml`

#### 2.2. CIFAR-10 (Natural Images)

- `torchvision.datasets.CIFAR10` handles downloading automatically.
- Data will be stored under `data/cifar10` as specified in `config.yaml`.

#### 2.3. PatchCamelyon (PCam, Histopathology)

- Uses `torchvision.datasets.PCAM` with train/val/test splits.
- The dataset will be downloaded to `data/pcam` automatically.

---

### 3. Architectures

All models are implemented in `models.py`, and selected via `model.name` in `config.yaml`.

- **Architecture 1 — MLP (`mlp`)**
  - `MLPClassifier`
  - Input: flattened features or images
  - Hidden layers: configurable through `model.hidden_dims`
  - Uses ReLU, optional BatchNorm, and Dropout
  - **Used on all three datasets**

- **Architecture 2 — CNN (`cnn`)**
  - `TabularCNN` for Adult:
    - Treats the feature vector as a 1D sequence and applies 1D convolutions
  - `ImageCNN` for CIFAR‑10 and PCam:
    - Stack of 2D conv + pooling + global pooling + MLP head
  - **Used on all three datasets**

- **Architecture 3 — Attention / Deep Feature (`attention`)**
  - `TabularAttentionMLP` for Adult:
    - Computes feature‑wise attention weights and re‑weights features before an MLP
  - `VisionTransformer` for CIFAR‑10 and PCam:
    - Patch embedding
    - Small Transformer encoder (2 layers, multi‑head self‑attention)
    - Classification head on `[CLS]` token
  - **Used on all three datasets (Adult uses attention‑MLP; image datasets use ViT‑style encoder)**

This satisfies the assignment requirement for an attention/deep‑feature model.

---

### 4. Configuration and Reproducibility

All experiment settings live in `config.yaml`:

- **Dataset selection**

```yaml
dataset:
  name: "adult"  # "adult", "cifar10", or "pcam"
```

- **Model selection and hyperparameters**

```yaml
model:
  name: "mlp"      # "mlp", "cnn", "attention"
  hidden_dims: [256, 128]
  dropout: 0.3
  use_batchnorm: true
```

- **Training configuration**

```yaml
training:
  batch_size: 512
  num_epochs: 20
  optimizer: "adam"
  lr: 0.001
  weight_decay: 0.0001
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.001
```

- **Logging / outputs**

```yaml
logging:
  output_dir: "outputs"
```

Each run writes a `results.json` file under `outputs/<experiment_name>/` containing:

- Training and validation loss / accuracy per epoch
- Final test metrics
- The full config used for that run

---

### 5. Running Experiments (9 Total)

#### 5.1. Single experiment

The main entrypoint is:

```bash
python run_experiment.py --config config.yaml
```

Edit `config.yaml` to choose the dataset/model pair and `experiment_name` for each run.

#### 5.2. All 9 experiments automatically

Alternatively, you can run all **3 datasets × 3 architectures** at once:

```bash
python run_all_experiments.py
```

This script loops over:

1. **Adult × MLP**          → `adult_mlp`
2. **Adult × CNN**          → `adult_cnn`
3. **Adult × Attention**    → `adult_attention`
4. **CIFAR‑10 × MLP**       → `cifar10_mlp`
5. **CIFAR‑10 × CNN**       → `cifar10_cnn`
6. **CIFAR‑10 × Attention** → `cifar10_attention`
7. **PCam × MLP**           → `pcam_mlp`
8. **PCam × CNN**           → `pcam_cnn`
9. **PCam × Attention**     → `pcam_attention`

Each experiment produces:

- `outputs/<experiment_name>/results.json` with:
  - Final test metrics
  - Training history (loss/accuracy per epoch)
  - Training time and parameter count
  - The full config used
- `loss_curves.png`, `accuracy_curves.png` for learning curves
- `conv_filters.png` (for CNN/ViT models with Conv2d layers)

---

### 6. Results Table Template

After running all experiments and collecting `results.json` files, you can create a summary table either manually **or automatically**:

- **Automatic summary**:

```bash
python summarize_results.py
```

This prints a Markdown table and generates parameter‑vs‑accuracy plots per dataset in the `outputs/` directory.

- **Manual template** (if you want to edit by hand):

| Dataset | Architecture     | Accuracy | F1     | Notes                              |
|--------|-------------------|----------|--------|------------------------------------|
| Adult  | MLP               |          |        |                                    |
| Adult  | CNN               |          |        |                                    |
| Adult  | Attention-based   |          |        |                                    |
| CIFAR10| MLP               |          |        |                                    |
| CIFAR10| CNN               |          |        |                                    |
| CIFAR10| Attention-based   |          |        |                                    |
| PCam   | MLP               |          |        |                                    |
| PCam   | CNN               |          |        |                                    |
| PCam   | Attention-based   |          |        |                                    |

Fill in the metrics from each `outputs/<experiment_name>/results.json`.

---

### 7. Analysis Report Outline (for GitHub README or separate report)

You can use the following outline for your analysis report (as requested in the assignment). This can either be expanded here in `README.md` or written as a separate `REPORT.md`:

- **Objective and Learning Outcomes**
  - Explain that the goal is to compare different inductive biases (MLP vs CNN vs attention/ViT) on different data modalities (tabular vs natural images vs histopathology images).
  - Mention that you focus on both **quantitative** results (accuracy, F1) and **qualitative** insights (which architectures are more suitable and why).

- **Code Structure and Modular Design**
  - `datasets.py`: dataset loading and preprocessing for Adult, CIFAR‑10, and PCam.
  - `models.py`: implementations of:
    - `MLPClassifier` (fully connected network)
    - `TabularCNN` and `ImageCNN`
    - `TabularAttentionMLP` and `VisionTransformer`
  - `trainer.py`: training loop, early stopping, and metric computation.
  - `run_experiment.py`: single entrypoint that reads a config and runs one experiment.
  - `config.yaml`: all hyperparameters and dataset/model selections.

- **Dataset Descriptions**
  - **Adult**: feature types (numeric + categorical), preprocessing (one‑hot, scaling).
  - **CIFAR‑10**: small 32×32 RGB images, 10 classes.
  - **PCam**: 96×96 histopathology patches, binary labels (tumor vs normal).

- **Architectures and Inductive Biases**
  - Why MLP is a natural baseline for tabular data but struggles with spatially structured images.
  - Why CNNs are more suited to images (translation invariance, local receptive fields).
  - Why attention/ViT‑style models can capture long‑range dependencies and global context, especially in images like PCam.

- **Results and Discussion**
  - For each dataset, compare:
    - Which architecture performs best and by how much.
    - Training dynamics (e.g., overfitting for MLP on CIFAR‑10 vs better generalization for CNN/ViT).
  - Discuss trade‑offs:
    - Parameter count vs performance
    - Training time vs accuracy
  - Explain any surprising findings or failures (e.g., attention underperforming CNN on small data, MLP overfitting tabular features).

- **Takeaways**
  - Summarize what you learned about:
    - Matching architectures to data modalities.
    - Importance of preprocessing and regularization.
    - Designing reproducible experiments with configs.

---

### 8. How This Maps to the Assignment Requirements

- **Code (50%)**
  - Clean, modular structure (`datasets.py`, `models.py`, `trainer.py`, `run_experiment.py`).
  - All important settings controlled through `config.yaml`.
  - Easily extensible to new architectures or datasets.

- **Results Table (15%)**
  - Automatically generated via `summarize_results.py`, or you can copy the Markdown table into your README/report and fill it in.

- **Analysis Report (35%)**
  - A complete `REPORT.md` template is provided with:
    - Objective and learning outcomes
    - Code structure
    - Dataset and architecture descriptions
    - Discussion of learning curves, parameter vs performance, and weight visualizations
  - You only need to plug in your actual numbers and any dataset‑specific insights you observe.

Finally, once you have run your experiments and filled in the results/analysis, you can:

- Initialize a git repository in this folder.
- Commit the code, `README.md`, `REPORT.md`, and (optionally) selected plots.
- Push to GitHub and submit the repository link as required by your assignment.

You can now push this project to GitHub with your results and expanded analysis to fully satisfy the assignment submission instructions.

