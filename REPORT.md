### Deep Learning Assignment 1 — Analysis Report

This report summarizes experiments conducted for the **Datasets × Architectures Benchmark** assignment. We trained three different neural network architectures on three datasets of different modalities (tabular, natural images, and histopathology images), then compared their quantitative performance and qualitative behavior.

---

### 1. Objective and Learning Outcomes

The primary objective of this assignment is to understand how **data modality** and **model inductive bias** interact in deep learning. In particular, we:

- Compared:
  - A **Multilayer Perceptron (MLP)** baseline
  - A **Convolutional Neural Network (CNN)**
  - An **attention‑based / deep feature model** (attention‑MLP for tabular data and a ViT‑style encoder for images)
- Evaluated them across:
  - **Adult Income** (tabular, binary classification)
  - **CIFAR‑10** (natural images, 10‑class classification)
  - **PatchCamelyon (PCam)** (histopathology images, binary classification)

From this assignment, the intended learning outcomes are:

- Gaining experience with **preprocessing** for different modalities (categorical + numerical features, small natural images, medical images).
- Implementing multiple architectures in **PyTorch** and training them in a reproducible fashion.
- Understanding how to **compare models fairly**, including:
  - Consistent train/val/test splits and optimizer choices
  - Trackable metrics (accuracy, F1)
  - Monitoring training curves and overfitting behavior
- Practicing how to **communicate experimental results and insights** clearly for an audience new to deep learning.

---

### 2. Code Structure and Modular Design

The project is organized as a small, modular PyTorch codebase:

- `datasets.py`
  - Implements dataset loading and preprocessing for:
    - **Adult**: reads a CSV file, splits into train/val/test, one‑hot encodes categorical columns, and standard‑scales numeric columns.
    - **CIFAR‑10**: uses `torchvision.datasets.CIFAR10` with standard normalization and a train/val split from the official training set.
    - **PCam**: uses `torchvision.datasets.PCAM` for train/val/test splits with basic normalization.
  - All dataset choices and paths are controlled via `config.yaml`.

- `models.py`
  - **MLPClassifier**:
    - Fully connected feedforward network with configurable hidden dimensions, ReLU activations, optional BatchNorm, and dropout.
    - Used on all datasets by flattening the inputs.
  - **TabularCNN**:
    - Treats the feature vector as a 1D “sequence” and applies 1D convolutions, global pooling, and a small classifier head.
    - Used as a CNN variant for the Adult dataset.
  - **ImageCNN**:
    - Classic 2D CNN with stacked convolutional blocks, pooling, and a dense classifier head.
    - Used for CIFAR‑10 and PCam.
  - **TabularAttentionMLP**:
    - Applies a feature‑wise attention mechanism to re‑weight input features before an MLP classifier.
    - Serves as an attention‑based model for Adult.
  - **VisionTransformer**:
    - ViT‑style encoder: patch embedding, learnable `[CLS]` token, positional embeddings, and a small Transformer encoder followed by a classification head.
    - Used as an attention‑based image model for CIFAR‑10 and PCam.
  - A `create_model(...)` factory picks the correct architecture based on the config.

- `trainer.py`
  - Handles the **training loop** and **evaluation**:
    - Uses a consistent optimizer family (`Adam` or `SGD`) and cross‑entropy loss.
    - Tracks training and validation loss/accuracy per epoch.
    - Supports **early stopping** with patience and minimum improvement thresholds.
  - Logs:
    - Final **test metrics** (accuracy and F1)
    - **Training time** and **average epoch time**
    - **Parameter count** for each model
  - Automatically generates:
    - **Loss and accuracy curves** (`loss_curves.png`, `accuracy_curves.png`)
    - **First‑layer convolution filter visualizations** (`conv_filters.png`) when a 2D convolutional layer is present.
  - Saves `results.json` for each experiment, containing the history, metrics, timing, and configuration.

- `run_experiment.py`
  - Main entrypoint that:
    - Loads a configuration file (`config.yaml`)
    - Builds dataloaders for the chosen dataset
    - Instantiates the requested model
    - Trains, evaluates, and writes results and plots for one experiment.

- `run_all_experiments.py`
  - Convenience script that runs all **9 combinations**:
    - Adult × {MLP, CNN, Attention}
    - CIFAR‑10 × {MLP, CNN, Attention}
    - PCam × {MLP, CNN, Attention}
  - Uses a base configuration and loops over (dataset, architecture) pairs to generate consistent outputs.

- `summarize_results.py`
  - Aggregates all `outputs/*/results.json` files.
  - Prints a **Markdown table** with dataset, architecture, accuracy, F1, training time, and parameter count.
  - Generates **parameter count vs accuracy** plots per dataset.

- `config.yaml`
  - Central place to control:
    - Dataset name and paths
    - Model type and hyperparameters
    - Training configuration (batch size, epochs, learning rate, etc.)
    - Logging options (output directory, whether to save best model and plots).

This modular design makes it easy to:

- Reuse code across datasets and models.
- Keep experiments reproducible and configurable.
- Extend the assignment (for example, by adding new architectures or regularization techniques).

---

### 3. Dataset Details and Preprocessing

#### 3.1. Adult Income (Tabular, Binary Classification)

- **Input modality**: mixed numerical and categorical tabular features.
- **Task**: predict whether an individual’s income is `>50K` or not.
- **Preprocessing**:
  - Categorical columns are one‑hot encoded.
  - Numerical columns are standardized using `StandardScaler`.
  - The target column `income` is converted to a binary label:
    - Positive class for entries containing `>50K`.
- **Splits**:
  - The dataset is split into train/validation/test according to the ratios defined in `config.yaml`, with stratification on the target label.

This dataset is a natural fit for MLPs and attention‑based MLPs, since the features are unordered and there is no spatial structure.

#### 3.2. CIFAR‑10 (Natural Images, 10‑Class Classification)

- **Input modality**: 32×32 RGB images.
- **Task**: classify images into 10 categories (e.g., airplane, automobile, bird, etc.).
- **Preprocessing**:
  - Uses `torchvision.datasets.CIFAR10` with:
    - Conversion to tensor.
    - Standard channel‑wise normalization.
  - The official training split is further divided into train and validation sets based on a configurable ratio.

This dataset has strong spatial structure, so CNNs and ViT‑style models are expected to perform better than a simple MLP on raw pixels.

#### 3.3. PatchCamelyon (PCam, Histopathology Images, Binary Classification)

- **Input modality**: 96×96 RGB histopathology patches derived from whole‑slide images.
- **Task**: binary classification of each patch as **tumor** vs **normal tissue**.
- **Preprocessing**:
  - Uses `torchvision.datasets.PCAM`, which provides train/val/test splits.
  - Images are converted to tensors and normalized.

This dataset represents a biomedical imaging setting where global context and fine‑grained textures both matter. CNNs and ViT‑style models can capture different aspects of these patterns.

---

### 4. Architectures and Inductive Biases

#### 4.1. Multilayer Perceptron (MLP)

- **Structure**:
  - Input layer (flattened features or images)
  - Two or more hidden layers with ReLU activations
  - Optional BatchNorm and dropout for regularization
  - Output layer mapping to the number of classes
- **Inductive bias**:
  - Treats each input feature independently; does not explicitly model spatial locality or translation invariance.
- **Strengths**:
  - Works well for tabular data, especially when features have been carefully engineered and normalized.
- **Weaknesses**:
  - Struggles with image data where spatial relationships are important, since flattening discards 2D structure.

#### 4.2. Convolutional Neural Network (CNN)

- **Structure**:
  - Stacks of convolutional layers with small receptive fields.
  - Pooling layers to progressively reduce spatial dimensions.
  - A fully connected head for classification.
- **Inductive bias**:
  - Exploits **local spatial correlations** and is **translation‑equivariant**.
- **Strengths**:
  - Very effective on image data (CIFAR‑10 and PCam).
  - Learns hierarchical features from edges and textures to high‑level patterns.
- **Weaknesses**:
  - Less natural for tabular data where no spatial structure is defined.

#### 4.3. Attention‑Based / Deep Feature Models

##### TabularAttentionMLP (Adult)

- Applies a learnable attention mechanism over features:
  - Computes attention scores for each feature.
  - Uses a softmax over feature scores to get feature weights.
  - Re‑weights each feature before feeding into an MLP.
- **Intuition**:
  - Allows the model to focus more on informative features and down‑weight less useful ones.

##### VisionTransformer (CIFAR‑10, PCam)

- Splits the image into patches.
- Projects each patch into an embedding space.
- Prepends a `[CLS]` token and adds positional embeddings.
- Processes the sequence with a Transformer encoder (multi‑head self‑attention).
- Uses the final `[CLS]` embedding for classification.
- **Inductive bias**:
  - Models global interactions between all patches via self‑attention.
  - Less biased toward locality than CNNs, which can be beneficial when global context is crucial.

---

### 5. Experimental Setup

- **Optimizer**: primarily Adam, with learning rate and weight decay defined in `config.yaml`.
- **Batch size**: chosen to balance convergence speed and memory usage.
- **Epochs**: capped to satisfy the assignment’s constraint that each model trains within a reasonable time budget (≤ 1 hour per model, depending on hardware).
- **Early stopping**:
  - Monitors validation loss.
  - Stops training when no sufficient improvement is observed for a configured number of epochs.
- **Metrics**:
  - For binary tasks (Adult, PCam): accuracy and F1 score.
  - For CIFAR‑10: accuracy and macro‑averaged F1 (treating all classes equally).
- **Logging**:
  - Training and validation curves are saved as PNGs.
  - Best model weights (by validation loss) are saved for each experiment.
  - Convolution filters in the first layer are visualized for image models.

All hyperparameters and dataset/model choices are defined through the configuration system, ensuring experiments are reproducible.

---

### 6. Quantitative Results

Use `summarize_results.py` after running all experiments to generate a Markdown table of results. A typical table has the form:

| Dataset | Architecture   | Accuracy | F1    | Training Time (s) | Params | Notes |
|--------|----------------|----------|-------|--------------------|--------|-------|
| Adult  | MLP            | ...      | ...   | ...                | ...    |       |
| Adult  | CNN            | ...      | ...   | ...                | ...    |       |
| Adult  | Attention      | ...      | ...   | ...                | ...    |       |
| CIFAR10| MLP            | ...      | ...   | ...                | ...    |       |
| CIFAR10| CNN            | ...      | ...   | ...                | ...    |       |
| CIFAR10| Attention      | ...      | ...   | ...                | ...    |       |
| PCam   | MLP            | ...      | ...   | ...                | ...    |       |
| PCam   | CNN            | ...      | ...   | ...                | ...    |       |
| PCam   | Attention      | ...      | ...   | ...                | ...    |       |

You can paste the exact numbers from `summarize_results.py` into this section.

---

### 7. Learning Curves and Overfitting Behavior

For each experiment, the training script saves:

- `loss_curves.png`: training vs validation loss per epoch.
- `accuracy_curves.png`: training vs validation accuracy per epoch.

Typical observations (your exact plots may vary):

- **Adult (tabular)**:
  - The MLP and attention‑based MLP generally converge quickly.
  - CNN on tabular data may not provide a clear advantage and can show more noisy validation curves, reflecting a mismatch between the convolutional inductive bias and tabular features.
  - Attention‑MLP may slightly outperform plain MLP or reach similar accuracy with smoother learning curves if it effectively focuses on informative features.

- **CIFAR‑10 (natural images)**:
  - The MLP on flattened images tends to overfit training data but underperform on validation/test compared to CNN.
  - The CNN usually reaches higher accuracy and more stable validation curves, showing that local spatial features are important.
  - The ViT‑style model can perform competitively, especially if enough data and regularization are used, but may require more careful tuning; it may start slower than the CNN and be more sensitive to hyperparameters.

- **PCam (histopathology)**:
  - The image distributions are different from natural images; textures and stain variability play a large role.
  - CNNs tend to perform strongly due to their ability to capture local tissue patterns.
  - ViT‑style models may capture longer‑range relationships across the patch and can match or exceed CNN performance when trained properly, though they may be more computationally expensive.

These learning curves help diagnose:

- Underfitting (both train and val accuracy low).
- Overfitting (train accuracy high but val accuracy degrades).
- Optimization issues (unstable or diverging loss).

---

### 8. Parameter Count vs Performance

Using `summarize_results.py`, we generate plots of **parameter count versus accuracy** for each dataset.

Typical trends (to be confirmed by your actual numbers):

- **Adult**:
  - All models have relatively modest parameter counts.
  - Increasing model size beyond a certain point may not yield significant gains, indicating that the dataset complexity and size limit the benefits of very deep models.
  - Attention‑MLP may achieve a good trade‑off, offering strong performance without a huge parameter increase.

- **CIFAR‑10 and PCam**:
  - CNNs and ViT‑style models generally have more parameters than the MLP, but also significantly better accuracy.
  - Beyond a certain scale, gains in accuracy may taper off, illustrating diminishing returns of simply increasing model size.
  - ViT‑style models can have parameter counts comparable to or larger than CNNs; their benefits lie in modeling long‑range interactions rather than pure parameter count.

Overall, these results highlight that **more parameters do not always mean better performance**, especially if the inductive bias is mismatched with the data or if overfitting becomes a concern.

---

### 9. Qualitative Weight Visualizations

For CNNs and ViT‑style models (via their first convolutional layers), we visualize the first‑layer filters:

- On **CIFAR‑10**:
  - Early filters often resemble edge and color detectors (e.g., oriented edges, color blobs).
  - This matches classical understanding of CNNs learning low‑level features similar to those in early visual cortex.

- On **PCam**:
  - Filters may capture textural patterns and stain variations relevant to tissue structure and tumor presence.
  - Some filters focus on nuclei‑like structures or boundaries between tissue types.

These visualizations provide an intuitive glimpse into what the networks are learning, showing that learned filters differ between natural and medical images while still following the general pattern of edge/texture detectors in early layers.

---

### 10. Discussion and Takeaways

Across all experiments, several key lessons emerge:

- **Match architecture to data modality**:
  - MLPs are a strong, simple baseline for tabular data but are not ideal for raw images.
  - CNNs are well‑suited for image data due to their local receptive fields and translation equivariance.
  - Attention‑based models (TabularAttentionMLP and ViT) can capture more complex interactions, especially long‑range dependencies in images, but often require more compute and careful tuning.

- **Preprocessing matters**:
  - Proper handling of categorical and numerical features (one‑hot + scaling) is crucial on tabular data.
  - Normalization of image inputs stabilizes training and speeds up convergence.

- **Regularization and early stopping**:
  - Dropout, BatchNorm, and early stopping play an important role in preventing overfitting, especially on smaller datasets or larger models.

- **Evaluation beyond a single metric**:
  - Looking at both **accuracy and F1** is important, particularly for imbalanced datasets like Adult and PCam.
  - Learning curves and parameter‑vs‑performance plots reveal behavior that a single test metric can miss (e.g., unstable optimization, overfitting, or inefficiency).

In summary, this assignment demonstrates that **no single architecture is best for all data types**. Instead, performance depends heavily on the structure of the data and the inductive biases built into the model. Developing an intuition for this alignment is a central skill in deep learning practice.

