model hahah


point to be noted 


1.u wil get very number of dataset


2. we are not allowd change the mode l archi ot model itself, or use any pretrained weigthts

Fixed model (ResNet-18)

steps 

1. we need 3lc account


2. collect the api key from the 3lc  for the initial setup

3.python version in b/w 3.9 and 3.14

code for env

```bash

python -m venv 3lc-env

source 3lc-env/bin/active

```

for GPU guys


1. Install PyTorch with GPU support (command depends on your CUDA version)
Example command from video for specific CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
 ```

 nvcc --version

 2.

 
Step 3: Install 3LC and dependencies (If you want little to no hassle use Option B)

Important - install order for GPU: PyTorch is installed by 3LC by default as CPU-only. If you want GPU training, install PyTorch with CUDA first, then 3LC (so 3LC uses the existing CUDA build). If you install 3LC first, pip install torch --index-url ... later will not replace the CPU build.

Option A - You have a GPU (for faster training):

1.Check your CUDA version (e.g. nvcc --version). If you have an Nvidia GPU and see an error like this when you run the above command: "'nvcc' is not recognized as an internal or external command, operable program or batch file." Install

a. Confirm the GPU

    Open Device Manager → Display adapters.

    If you see an NVIDIA card (e.g. GeForce, RTX), you have an NVIDIA GPU.

    b. Install CUDA:

    Download the CUDA Toolkit from NVIDIA CUDA Toolkit.

    Run the installer; it usually adds CUDA to PATH.

    Restart the terminal (or PC if needed), then run nvcc --version again.

2.Install PyTorch with the matching index before installing 3LC:

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

See PyTorch Get Started for other CUDA versions.

3.Install 3LC and extras (do not install torch again here):

pip install 3lc joblib pytz umap-learn

4.Verify GPU (recommended):

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

You should see CUDA: True and your GPU name. If you see CUDA: False, you have the CPU-only build; install PyTorch with the CUDA index above before 3LC (see troubleshooting below if you already installed 3LC).

Option B - CPU only (slightly slower): Recommended if you don't want the headache of the GPU support installation if you don't have it already

pip install 3lc joblib pytz umap-learn torch torchvision

If you face

    umap-learn is used for embeddings reduction (3D in the Dashboard). See 3LC FAQ - Dimensionality reduction.

Option C - macOS (Apple Silicon): If the standard pip install 3lc fails on Mac, use the wheel from PyPI:

    Download the appropriate wheel from https://pypi.org/project/3lc/#files (select by Python version and macOS).
    Example for Python 3.13 on macOS ARM64: 3lc-2.22.0-cp313-cp313-macosx_13_0_arm64.whl
    Install Python 3.13 if needed: brew install python@3.13
    Create venv: python3.13 -m venv 3lc-env then source 3lc-env/bin/activate
    Upgrade pip: python -m pip install --upgrade pip
    Install the wheel: python -m pip install ~/Downloads/3lc-2.22.0-cp313-cp313-macosx_13_0_arm64.whl (adjust path and filename to match your download)
    Install rest: pip install joblib pytz umap-learn torch torchvision
    Log in to 3LC using the instructions below.

Step 4: Log in to 3LC

In the same terminal (with your environment activated):

3lc login <your_api_key>

Use the API key from https://account.3lc.ai/api-key. This saves the key locally; you don't need to run this again in future sessions.
Step 5: Start the 3LC service (for Dashboard)

    The 3LC service is not required for training - you can run train.py without it.
    It is required to use the 3LC Dashboard (visualize embeddings, label undefined, inspect runs). For this competition's data-centric loop, we recommend running it.

In a terminal:

3lc service

    Starts the local Object Service so the Dashboard can talk to your data.
    Your data stays on your machine and is not uploaded to 3LC. See 3LC FAQ – Do I need to upload my data?
    Keep this terminal open while using the Dashboard. To stop: press Q or Ctrl+C.

Step 6: Open the 3LC Dashboard

With the service running, open https://dashboard.3lc.ai in your browser.

    First-time users: Follow the on-screen tour to learn the interface.
    Browsers: Chrome (recommended), Firefox, or Edge (latest). For smoother experience, enable hardware acceleration: 3LC Dashboard GPU Acceleration.
    If the Dashboard can't connect, see 3LC FAQ – Why can't the Dashboard connect?

Verification checklist

Before running the workflow, confirm:
Check 	Action
3LC account 	Created; API key from account.3lc.ai/api-key
Environment 	Venv (or chosen env) activated
PyTorch GPU 	If using GPU: PyTorch installed with CUDA before 3LC
3LC installed 	pip install 3lc joblib pytz umap-learn (and torch/torchvision if CPU)
Logged in 	3lc login <api_key> done once
Dashboard (optional) 	3lc service running; dashboard.3lc.ai open

Future sessions: Activate environment → (optional) start 3lc service and open Dashboard.
Environment troubleshooting
Issue 	What to do
3lc: command not found 	Activate your Python environment and try again or reinstall 3lc
API key invalid 	Get a fresh key at account.3lc.ai/api-key
No GPU detected during training 	Install PyTorch with CUDA before 3LC; see Step 3 Option A
Cannot connect to Dashboard 	Ensure 3lc service is running in a terminal; see FAQ - Dashboard connection
Additional resources

    3LC Documentation
    3LC Example Notebooks
    3LC Getting Started Video Playlist
    Hackathon videos: Part 1 · Part 2 · Part 3
    Support: Discord

Iterative Workflow with 3LC - The Main Sauce

Video walkthrough: Part 2 - First Run & Submit · Part 3 - Weighting Strategies
Iterative workflow - What you're proving

    Train on few labeled examples → model is weak.
    Get predictions on all unlabeled data (and see 3D embeddings in 3LC).
    Use embeddings as model feedback to label some unlabeled data and add it to the training set in 3LC.
    Retrain (3LC uses the new table automatically) → better model.
    Repeat a few times, then predict on test and submit.

Concepts you'll use
Concept 	Meaning in this competition
Table 	A 3LC dataset (e.g. train or val). Has rows = samples and columns = image path, label, weight, etc.
Revision 	A version of a table. Editing labels/weights in the Dashboard creates a new revision; train.py uses the latest.
Run 	One training run. 3LC stores metrics and embeddings per run so you can inspect them in the Dashboard.
Embeddings 	3D points from the model's features (UMAP). Used to find clusters (e.g. chihuahua vs muffin) and pick which undefined to label.
Sample weight 	Rows with weight=1 are used in training; weight=0 (e.g. undefined until you label them) are skipped.
Mechanics (how it fits together)

    Register once: register_tables.py creates 3LC tables from data/train and data/val (project/dataset/table names are in the script).
    Train: train.py loads tables with .latest(), so it always uses the newest revision (including Dashboard edits).
    Edit in Dashboard: When you label undefined samples and set weight=1, 3LC creates a new table revision. Next train.py run uses that revision automatically.
    Test is not registered; predict.py reads images from data/test/ and outputs submission.csv.

Step 1: Register data in 3LC (once, ~2 min)

NOTE: Activate your '3lc-env' and 'cd' into the starter kit directory that has the following files

What: Create 3LC tables from data/train and data/val. Test is not registered.

Run:

python register_tables.py

You should see: Messages like "Created 3LC table: train", "Created 3LC table: val", and table URLs. No errors.

If it says tables already exist: That's fine. Use the existing tables.
Step 2: First training run: train on labeled only (~2 min)

What: Train the model on the current table. Only labeled rows (chihuahua, muffin) have weight=1; undefined have weight=0, so they are not used yet.

Run:

python train.py

You should see: Epochs running, validation accuracy at the end (e.g. 60–75%). At the end it saves best_model.pth and writes metrics/3D embeddings to 3LC. The starter uses UMAP for embeddings.

In the 3LC Dashboard: Open the run you just created. You'll see embeddings (points in 3D) and per-sample metrics (e.g. loss, predicted label, confidence). This is "model feedback" on the training set.

Aha: With only 100 labeled images, the model is limited. Many undefined images are unlabeled; we'll use the model's predictions and embeddings to choose which to label next.
Step 3: Use embeddings to label some undefined (~5–10 min)

What: In 3LC, use the run you just created to see which undefined samples the model is confident about, then label a batch and add them to the training set.

In the 3LC Dashboard:

    Open the Tables view and select the train table (latest version).
    Open the Run you created in Step 2 so you see embeddings and metrics for that table.
    Filter to show only rows where label = undefined (or "unlabeled", depending on UI).
    Use embeddings: Select the embeddings column and press "3" on your keyboard to create a 3D embeddings chart.
        Find clusters of points. Clusters often correspond to one class (e.g. chihuahua vs muffin). You can color the embeddings with class, confidence or loss by dragging the respective column on where it is written LABEL on the top of the embeddings chart.
        Optionally sort or filter by confidence (model's prediction confidence) to pick low-confidence undefined samples.
    Label a batch: Select a set of undefined samples (e.g. 50–100) that look like one class in the embedding view, and set their label to chihuahua or muffin.
    Set their weight to 1 so they are used in training.
    Save / commit a new table version (3LC will create a new revision).

Result: Your train table now has more labeled rows (e.g. 100 + 100 = 200). The new revision is what train.py will use next time (it uses .latest()).

Aha: You didn't label randomly, you used model feedback (embeddings and confidence) to choose which unlabeled samples to add. That's the data-centric loop.
Step 4: Retrain (~2–5 min)

What: Train again. The script loads the latest table, so your newly labeled samples are included.

Run:

python train.py

You should see: A new run in 3LC, and usually higher validation accuracy (e.g. 75%) because you added good labeled data.

Optional: Repeat Step 3 and Step 4 once more: label another batch of undefined using embeddings, then run python train.py again. Each iteration can improve the model.
Step 5: Predict on test and submit (~2 min)

What: Run the saved best model on the test images and create the submission file for Kaggle.

Run:

python predict.py

You should see: "Predicting…" over data/test/, then "Written to submission.csv".

Check: Open submission.csv. It must have columns image_id, prediction, confidence and one row per test image (same image_ids as in sample_submission.csv).

Submit: Upload submission.csv to the Kaggle competition. Your score is accuracy on the hidden test set. Better data (from 3LC) → better accuracy.
Quick reference (copy-paste order)

# Once
3lc login YOUR_API_KEY
3lc service   # whenever you want to access the Dashboard
python register_tables.py

# First train
python train.py

# Then in 3LC Dashboard: label some undefined using embeddings, set weight=1, save new table version

# Retrain (repeat "label in Dashboard" + "train" if you have time)
python train.py

# Submit
python predict.py
# Upload submission.csv to Kaggle

Troubleshooting
Problem 	What to do
register_tables.py says "Data directory not found" 	Make sure you're in the folder that contains data/train, data/val, data/test.
train.py fails or "table not found" 	Run python register_tables.py first. If it says "tables already exist", use the same project/dataset in 3LC that the script uses.
Dashboard blank or won't load 	Keep 3lc service running; use Chrome or Edge; allow WebGL.
"No images in data/test" 	You need the competition dataset with test images (e.g. test_00001.jpg, …). Get it from organizers or Kaggle data.
submission.csv has wrong number of rows 	It must have exactly the same image_ids as sample_submission.csv. If your test folder has fewer images, get the full dataset.
I only have limited hours 	Do Step 1 → Step 2 → Step 3 (label at least 50–100 undefined) → Step 4 → Step 5. One full "label + retrain" loop is enough to see the improvement.

Summary: Train on few labels → use 3LC to get predictions and 3D embeddings on unlabeled data → label a batch using that feedback → retrain → predict on test and submit.

Support: Discord - #hackathons
FAQs
3LC and deployment

Q Do I need to upload my data to 3LC? No. The Dashboard talks to your local Object Service (3lc service). All data is created, viewed, and modified on your machine and is never sent to 3LC. Only authentication and loading the Dashboard UI use the internet. See 3LC FAQ – Do I need to upload my data?

Q Why can't the Dashboard connect to the Object Service?

    Ensure 3lc service is running and you open the URL it prints.
    If the service is not on default localhost:5015, set the connection URL in Dashboard Settings → CONNECTION.
    Use a supported browser (Chrome or Edge). If using Brave, turn off shields and allow local network access.
    See 3LC FAQ – Why can't the Dashboard connect?

Dashboard and images

Q Why can't I see my images in the Dashboard?

    Images are stored as paths (URLs). In the Dashboard, use "To string" on the image column to see the paths.
    If you use path aliases, the Object Service must be started with the same config that defines those aliases.
    Ensure the Object Service has read access to the image directories (or cloud credentials if data is remote).
    See 3LC FAQ – Why can I not see my images?

Q Why are my image transforms not applied in the Dashboard? 3LC persists untransformed samples (e.g. original PIL images) so the Dashboard shows the same data you'd use for debugging. Transforms are still applied when you iterate the table in code and during training. See 3LC FAQ – Why are my image transforms not applied?
Tables and naming

Q What are project_name, dataset_name, and table_name? They define where the table lives and how it's identified. Use them to organize and share data. Suggested: project_name = goal (e.g. "ChihuahuaMuffin"), dataset_name = split (e.g. "train", "val"), table_name = revision description (e.g. "initial", "added_labels_batch1"). See 3LC FAQ – What should be the project_name, dataset_name and table_name?

Q How do I add an extra column to a table? Use Table.add_column() to create a new revision with an extra column, or pass extra_columns when creating the table (e.g. in Table.from_torch_dataset). See 3LC FAQ – How can I add an extra column?

Q How do I export data from a table? Use 3LC's export APIs (e.g. to CSV or other formats). See the Export section in the 3LC user guide.
Object Service

Q Why doesn't the Object Service start? Check the console for errors. Often another process is using the same port (default 5015). Stop the other process or configure 3LC to use a different port. See 3LC FAQ – Why doesn't the Object Service start?
Competition-specific

Q Why does train.py use only labeled rows? Only rows with weight = 1 are used in the loss. Undefined rows start with weight 0; once you label them and set weight=1 in the Dashboard, the next training run includes them.

Q How many times should I do the "label → retrain" loop? One full loop (Steps 3 → 4) is enough to see the effect. If you have time, do two: label a first batch (e.g. 50–100), retrain, then label another batch and retrain again.

Q Is it 3LC, 3lc, or tlc? The product is 3LC. The Python package is installed as 3lc; you import tlc in code (because names can't start with a number). See 3LC FAQ – Is it spelled 3LC, 3lc or tlc?
Reference: Concepts

    Table: A 3LC dataset with a schema (columns and types). Created from folders or PyTorch datasets via register_tables.py / Table.from_* APIs.
    Revision: Tables are immutable; edits create a new revision. train.py uses .latest() so it always picks the newest revision.
    Run: One training run. 3LC records metrics (loss, accuracy, predictions) and embeddings for each run so you can inspect them in the Dashboard.
    Embeddings: Reduced (e.g. UMAP) feature vectors from the model. The starter uses UMAP for 3D embeddings. Used to visualize and cluster samples and to choose which unlabeled samples to label.
    Sample weight: Per-row weight used in the loss. Weight 0 = excluded from training; weight 1 = included. Undefined samples start at 0 until you label and enable them.
    Registration: register_tables.py builds 3LC tables from data/train and data/val and writes them to the 3LC project (path/config in script). Idempotent: safe to run again; existing tables are reused.
    Training: train.py loads train/val via tlc.Table.from_names(...).latest(), so Dashboard edits (new revisions) are used automatically on the next run. Metrics and 3D embeddings are written to a new run each time.
    Prediction: predict.py reads images from data/test/, runs the saved best model, and writes submission.csv with image_id, prediction, confidence for Kaggle.


