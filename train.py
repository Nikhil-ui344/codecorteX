"""
Improved training pipeline for Chihuahua vs Muffin (3LC competition)

Changes:
- ResNet18 from scratch (rule compliant)
- 224 image size
- better augmentations
- mixed precision training
- cosine learning rate scheduler
- faster dataloading
- improved embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import os

# ============================================================================
# CONFIG
# ============================================================================

EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
RANDOM_SEED = 42

PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

NUM_CLASSES = 2
CLASS_NAMES = ["chihuahua", "muffin", "undefined"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print("ResNet18 training from scratch (competition rule)")

# ============================================================================
# SEED
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    ),
])

def train_fn(sample):
    image = Image.open(sample["image"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return train_transform(image), sample["label"]

def val_fn(sample):
    image = Image.open(sample["image"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return val_transform(image), sample["label"]

# ============================================================================
# METRICS
# ============================================================================

def metrics_fn(batch, predictor_output: tlc.PredictorOutput):

    labels = batch[1].to(device)
    predictions = predictor_output.forward

    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)

    confidence = torch.gather(
        softmax_output,
        1,
        predicted_indices.unsqueeze(1)
    ).squeeze(1)

    accuracy = (predicted_indices == labels).float()

    valid_labels = labels < predictions.shape[1]

    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)

    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(
        reduction="none"
    )(predictions[valid_labels], labels[valid_labels])

    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }

# ============================================================================
# TRAIN
# ============================================================================

BEST_MODEL_FILENAME = "best_model.pth"

def train():

    base_path = Path(__file__).parent

    tlc.register_project_url_alias(
        token="CHIHUAHUA_MUFFIN_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )

    print("Loading tables...")

    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train",
    ).latest()

    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="val",
    ).latest()

    print("Train samples:", len(train_table))
    print("Val samples:", len(val_table))

    class_names = list(train_table.get_simple_value_map("label").values())

    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)

    train_sampler = train_table.create_sampler(exclude_zero_weights=True)

    train_loader = DataLoader(
        train_table,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_table,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler()

    run = tlc.init(
        project_name=PROJECT_NAME,
        description="Improved training run"
    )

    indices_and_modules = list(enumerate(model.resnet.named_modules()))

    resnet_fc_layer_index = next(
        (i for i,(n,_) in indices_and_modules if n=="fc"),
        len(indices_and_modules)-1
    )

    predictor = tlc.Predictor(model, layers=[resnet_fc_layer_index])

    classification_metrics_collector = tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn
    )

    embeddings_metrics_collector = tlc.EmbeddingsMetricsCollector(
        layers=[resnet_fc_layer_index]
    )

    best_val_accuracy = 0.0
    best_model_state = None

    print("Starting training...")

    for epoch in range(EPOCHS):

        model.train()

        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                preds = model(images).argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()

        tlc.log({"epoch": epoch, "val_accuracy": val_acc})

    print("Best validation accuracy:", best_val_accuracy)

    if best_model_state:
        model.load_state_dict(best_model_state)

    model_path = base_path / BEST_MODEL_FILENAME
    torch.save(model.state_dict(), model_path)

    print("Best model saved:", model_path)

    print("Collecting metrics...")

    tlc.collect_metrics(
        train_table,
        predictor=predictor,
        metrics_collectors=[
            classification_metrics_collector,
            embeddings_metrics_collector
        ],
        split="train",
        dataloader_args={
            "batch_size": BATCH_SIZE,
            "num_workers": 6
        },
    )

    print("Reducing embeddings...")

    run.reduce_embeddings_by_foreign_table_url(
        train_table.url,
        method="umap",
        n_neighbors=30,
        min_dist=0.05,
        n_components=3,
    )

    run.set_status_completed()

    print("Training completed.")

if __name__ == "__main__":
    train()