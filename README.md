# Model for Medal

A chill guide for the Chihuahua vs Muffin challenge using 3LC.
## Ground Rules (Keep It Fair)

- You receive a large dataset.
- You must keep the model architecture fixed.
- You must not use pretrained weights.
- Base model: ResNet-18.

Short version: improve data quality (labels + weights), not the architecture.
## Vibe Check: What You Need

- A 3LC account.
- An API key from https://account.3lc.ai/api-key.
- Python 3.9 to 3.14.
- Optional: NVIDIA GPU + CUDA for faster training.

## Setup 

### 1) Create and activate virtual environment

```bash
python -m venv 3lc-env
source 3lc-env/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

Important install order for GPU users:
Install CUDA-enabled PyTorch first, then install 3LC. If you install 3LC first, it may pull CPU-only PyTorch.

#### Option A: GPU (recommended)

1. Check CUDA availability:

```bash
nvcc --version
```

If `nvcc` is missing:

- Confirm you have an NVIDIA GPU.
- Install CUDA Toolkit from NVIDIA.
- Restart terminal and check again.

2. Install PyTorch matching your CUDA version:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8 (nightly)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

3. Install 3LC and extra dependencies:

```bash
pip install 3lc joblib pytz umap-learn
```

4. Verify GPU PyTorch build:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output should include `CUDA: True` and your GPU name.

#### Option B: CPU only (easy mode)

```bash
pip install 3lc joblib pytz umap-learn torch torchvision
```

#### Option C: macOS (Apple Silicon)

If `pip install 3lc` fails on macOS, install from PyPI wheel:

1. Download wheel from https://pypi.org/project/3lc/#files.
2. Install Python 3.13 if needed:

```bash
brew install python@3.13
```

3. Create env and activate:

```bash
python3.13 -m venv 3lc-env
source 3lc-env/bin/activate
```

4. Install wheel (example file name):

```bash
python -m pip install ~/Downloads/3lc-2.22.0-cp313-cp313-macosx_13_0_arm64.whl
```

5. Install remaining dependencies:

```bash
pip install joblib pytz umap-learn torch torchvision
```

## Login + Dashboard

### 3) Log in to 3LC

```bash
3lc login <your_api_key>
```

You only need this once per machine and environment.

### 4) Start 3LC Object Service (required for Dashboard)

```bash
3lc service
```

Notes:

- Training scripts can run without this.
- Dashboard features (tables, embeddings, labeling) need this running.
- Keep this process running while using Dashboard.
- Stop with `Q` or `Ctrl+C`.

### 5) Open Dashboard

Open https://dashboard.3lc.ai in Chrome, Edge, or Firefox.

## Starter Kit Scripts

This workflow assumes these scripts exist in your project:

- `register_tables.py`: registers `data/train` and `data/val` to 3LC.
- `train.py`: trains ResNet-18 using latest 3LC table revision.
- `predict.py`: runs inference on `data/test` and writes `submission.csv`.

## Main Sauce: Iterative Workflow

The loop is simple:

1. Train on current labeled data.
2. Inspect embeddings and predictions in 3LC.
3. Label high-value `undefined` samples.
4. Set labeled samples weight to `1`.
5. Retrain.
6. Repeat.

### Step 1: Register tables (once)

```bash
python register_tables.py
```

Expected result:

- 3LC `train` and `val` tables are created (or reused if they already exist).
- Table URLs are printed.

### Step 2: First training run

```bash
python train.py
```

Expected result:

- Epoch logs and validation accuracy printed.
- `best_model.pth` saved.
- Metrics and embeddings logged to a 3LC run.

### Step 3: Label `undefined` using embeddings (Dashboard)

In Dashboard:

1. Open latest `train` table.
2. Open the latest run for that table.
3. Filter rows where label is `undefined`.
4. View embeddings in 3D (press `3` on embeddings column view).
5. Pick a batch (for example 50 to 100 samples) and assign label.
6. Set `weight = 1` for newly labeled rows.
7. Save changes to create a new table revision.

### Step 4: Retrain on updated revision

```bash
python train.py
```

Because training uses `.latest()`, your new labels are used automatically.

### Step 5: Predict and create submission

```bash
python predict.py
```

Expected result:

- `submission.csv` written.
- Required columns: `image_id`, `prediction`, `confidence`.

Upload `submission.csv` to Kaggle.

## Quick Run Order

```bash
# one-time auth
3lc login YOUR_API_KEY

# dashboard session (optional but recommended)
3lc service

# one-time registration
python register_tables.py

# first train
python train.py

# dashboard: label undefined + set weight=1 + save new revision

# retrain (repeat loop if time allows)
python train.py

# generate submission
python predict.py
```

## Fast Checklist

| Check | Confirm |
|---|---|
| 3LC account | Created and API key available |
| Environment | Virtual environment active |
| GPU setup | CUDA PyTorch installed before 3LC (if GPU path) |
| 3LC install | `3lc`, `joblib`, `pytz`, `umap-learn` installed |
| Login | `3lc login` completed |
| Dashboard | `3lc service` running before opening dashboard |

## Troubleshooting

| Problem | Fix |
|---|---|
| `3lc: command not found` | Activate venv and reinstall `3lc` if needed |
| Invalid API key | Regenerate key at account.3lc.ai |
| GPU not detected | Reinstall CUDA-enabled PyTorch first, then reinstall 3LC deps |
| Dashboard cannot connect | Ensure `3lc service` is running; check browser and local network permissions |
| `Data directory not found` | Run scripts from project root containing `data/train`, `data/val`, `data/test` |
| Table not found in training | Run `python register_tables.py` first |
| Wrong submission row count | Ensure test folder matches official dataset and IDs |

## Core Concepts (Quick)

- Table: structured dataset in 3LC.
- Revision: immutable version created after edits.
- Run: one training execution with logged metrics/embeddings.
- Embeddings: reduced feature vectors (for example UMAP 3D).
- Sample weight: `1` included in training, `0` excluded.

## FAQ

### Do I need to upload data to 3LC?

No. The Dashboard connects to your local Object Service (`3lc service`). Data remains on your machine.

### Why are transforms not visible in Dashboard?

Dashboard usually shows untransformed stored samples for inspection. Transforms are still applied in training/data loading code.

### Is it 3LC, 3lc, or tlc?

- Product name: `3LC`
- Python package install command: `pip install 3lc`
- Python import in code: `import tlc`

## Limited Time? Do This

1. Register tables.
2. Train once.
3. Label 50 to 100 `undefined` samples using embeddings.
4. Retrain.
5. Predict and submit.

One full label-and-retrain cycle is usually enough to see improvement.

## Resources

- 3LC Documentation
- 3LC Example Notebooks
- 3LC Getting Started Videos
- Hackathon videos (Part 1, Part 2, Part 3)
- Discord support channel
