---
# **AI-Powered Sign Language Translation Project**

Welcome! 👋 This repository contains a **modular framework for Sign Language Translation** (Sign → Text and Text → Sign) using **landmarks and deep learning models**. Currently, we have a **TCN model**, but you can easily add new models like Transformers or LSTMs.

---

## **📂 Project Structure**

```
AI/
├─ src/
│  ├─ configs/                   # Model hyperparameters
│  │  ├─ tcn_config.py           # TCN-specific config
│  │  └─ transformer_config.py   # Example template for new models
│  ├─ data/
│  │  ├─ dataloaders.py          # Prepares PyTorch DataLoaders
│  │  └─ preprocessing/          # Landmark extraction & preprocessing scripts
│  ├─ models/
│  │  └─ sign2text/
│  │     ├─ tcn.py               # TCN model
│  │     ├─ losses.py            # Loss functions (e.g., SmoothCE)
│  │     └─ transformer.py       # Example template for a new model
│  ├─ training/
│  │  ├─ train_tcn.py            # TCN training script
│  │  ├─ train_transformer.py    # Example template for new models
│  │  └─ trainer.py              # Generic training utilities (run_epoch, etc.)
│  └─ evaluation/
│     ├─ eval_tcn.py             # TCN evaluation, metrics, confusion, save plots
│     └─ eval_transformer.py     # Example template for new models
├─ experiments/
│  └─ plots/                     # Saved metrics, confusion matrices, model structures
├─ requirements.txt               # Python dependencies
└─ README.md
```

---

## **📦 Environment Setup**

1. Clone the repo:

```bash
git clone <repo-url>
cd AI/
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **🗂️ Data Preparation**

* Place **landmark `.npy` files** in a folder (e.g., `Top_Classes_Landmarks_Preprocessed`)
* Preprocessing scripts are in: `src/data/preprocessing/`
* Data loaders are handled in `src/data/dataloaders.py`. Just call:

```python
from src.data.dataloaders import build_dataloaders
train_loader, val_loader, test_loader, num_classes = build_dataloaders()
```

---

## **⚡ Training a Model**

### **TCN Model (default)**

```bash
python -m src.training.train_tcn
```

* Saves checkpoints, best model, metrics automatically.
* Checkpoints: `MODEL_SAVE_PATH` defined in `tcn_config.py`

---

### **Adding a New Model (e.g., Transformer)**

1. **Create a model file**

```text
src/models/sign2text/transformer.py
```

* Implement a class like:

```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        ...
    def forward(self, x, mask):
        ...
        return out
```

2. **Create a config file**

```text
src/configs/transformer_config.py
```

* Define hyperparameters: `LR`, `BATCH_SIZE`, `EPOCHS`, `DEVICE`, `MODEL_SAVE_PATH`, etc.

3. **Create a training script**

```text
src/training/train_transformer.py
```

* Import your model and config
* Reuse: `build_dataloaders()`, `SmoothCE`, `run_epoch()`
* Run training:

```bash
python -m src.training.train_transformer
```

4. **Create an evaluation script**

```text
src/evaluation/eval_transformer.py
```

* Import your model and config
* Copy `eval_tcn.py` logic
* Replace TCN references → your model
* Use a `_transformer` suffix for saved metrics and plots
* Run evaluation:

```bash
python -m src.evaluation.eval_transformer
```

---

## **🛠 Reusable Components**

* **Data loaders** → `build_dataloaders()`
* **Loss function** → `SmoothCE`
* **Training loop** → `run_epoch()`
* **Evaluation utilities** → confusion matrix, metrics, plots

> You only need to implement the **model** and **its config**, everything else is reusable.

---

## **📊 Outputs**

For each model, the following are automatically saved:

* `experiments/plots/test_metrics_<model>.json` → evaluation metrics
* `experiments/plots/confusion_matrix_<model>.png` → confusion matrix
* `experiments/plots/model_structure_<model>.txt` → model structure

> Replace `<model>` with `_tcn`, `_transformer`, etc.

---

## **💡 Tips**

* Always give each model **unique checkpoint and plot filenames**.
* Keep configs separate per model for clarity.
* Later, you can merge training/eval scripts into **generic scripts** for multiple models.

---
Perfect, Mariam 😎 — here’s a simple **diagram you can add to the README** showing what your friends need to **create vs reuse** when adding a new model. You can place it under a section like **“Adding a New Model”**.

---

## **🖼️ Visual Guide: Adding a New Model**

```text
┌───────────────────────────────┐
│       Create for new model     │
├───────────────────────────────┤
│ src/models/sign2text/<model>.py   <- Your model class
│ src/configs/<model>_config.py     <- Hyperparameters & paths
│ src/training/train_<model>.py     <- Training script
│ src/evaluation/eval_<model>.py    <- Evaluation script
└───────────────────────────────┘

           ⬇ Reuse existing components

┌───────────────────────────────┐
│           Reusable             │
├───────────────────────────────┤
│ src/data/dataloaders.py       <- Data loaders
│ src/models/losses.py          <- SmoothCE and other losses
│ src/training/trainer.py       <- run_epoch(), gradient clipping, etc.
│ src/data/preprocessing/       <- Landmark extraction, sliding window
│ src/evaluation/utils.py       <- Confusion, metrics, plotting
└───────────────────────────────┘

```

**How it works:**

1. **Create** files for your new model, config, training, and evaluation.
2. **Reuse** everything else: data loaders, loss, training utilities, and evaluation helpers.
3. **Run training & evaluation** using your new scripts.
4. **Outputs** are automatically saved in `experiments/plots/` with a model-specific suffix.

---
