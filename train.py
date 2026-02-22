"""
M1: Model Development & Experiment Tracking
- Downloads Cats vs Dogs dataset from Kaggle
- Preprocesses images to 224x224 RGB
- Trains a simple CNN with PyTorch
- Logs everything to MLflow
- Saves the trained model
"""

import os
import shutil
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import kagglehub

# ─────────────────────────────────────────────
# 0. CONFIG  (change anything here freely)
# ─────────────────────────────────────────────
KAGGLE_DATASET   = "bhavikjikadara/dog-and-cat-classification-dataset"   # kaggle
DATA_DIR         = "data"
RAW_ZIP          = os.path.join(DATA_DIR, "dogs-vs-cats.zip")
TRAIN_DIR        = os.path.join(DATA_DIR, "train")
PROCESSED_DIR    = os.path.join(DATA_DIR, "processed")

IMAGE_SIZE       = 224
BATCH_SIZE       = 32
NUM_EPOCHS       = 5
LEARNING_RATE    = 0.001
TRAIN_SPLIT      = 0.80
VAL_SPLIT        = 0.10
# TEST_SPLIT     = 0.10  (remainder)

MODEL_SAVE_PATH  = os.path.join("models", "cats_dogs_cnn.pt")
SEED             = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 1. DOWNLOAD DATA FROM KAGGLE
# ─────────────────────────────────────────────
def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PROCESSED_DIR):
        print("Processed data already exists — skipping download.")
        return

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    # Copy downloaded files into DATA_DIR so prepare_data() finds them
    for name in os.listdir(path):
        src = os.path.join(path, name)
        dst = os.path.join(DATA_DIR, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print("Dataset ready in", DATA_DIR)

# ─────────────────────────────────────────────
# 2. ORGANISE INTO train / val / test FOLDERS
# ─────────────────────────────────────────────
def prepare_data():
    if os.path.exists(PROCESSED_DIR):
        print("Processed directory already exists — skipping split.")
        return

    # Locate raw images — the dataset keeps them in data/train/
    raw_dir = os.path.join(DATA_DIR, "train")
    if not os.path.isdir(raw_dir):
        # Some versions extract directly into DATA_DIR
        raw_dir = DATA_DIR

    all_images = [f for f in os.listdir(raw_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    cats = [f for f in all_images if f.startswith("cat")]
    dogs = [f for f in all_images if f.startswith("dog")]

    print(f"Found {len(cats)} cat images and {len(dogs)} dog images.")

    random.shuffle(cats)
    random.shuffle(dogs)

    def split_list(lst):
        n = len(lst)
        t = int(n * TRAIN_SPLIT)
        v = int(n * VAL_SPLIT)
        return lst[:t], lst[t:t+v], lst[t+v:]

    cat_train, cat_val, cat_test = split_list(cats)
    dog_train, dog_val, dog_test = split_list(dogs)

    splits = {
        "train": {"cat": cat_train, "dog": dog_train},
        "val":   {"cat": cat_val,   "dog": dog_val},
        "test":  {"cat": cat_test,  "dog": dog_test},
    }

    for split, classes in splits.items():
        for cls, files in classes.items():
            dest = os.path.join(PROCESSED_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            for fname in files:
                shutil.copy(os.path.join(raw_dir, fname), os.path.join(dest, fname))

    print("Data split complete:")
    for split, classes in splits.items():
        for cls, files in classes.items():
            print(f"  {split}/{cls}: {len(files)} images")


# ─────────────────────────────────────────────
# 3. DATA LOADERS
# ─────────────────────────────────────────────
def get_data_loaders():
    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure all images are RGB
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "val"),   transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "test"),  transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Classes: {train_dataset.classes}")   # ['cat', 'dog']
    return train_loader, val_loader, test_loader, train_dataset.classes


# ─────────────────────────────────────────────
# 4. SIMPLE CNN MODEL
# ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 28x28

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 14x14
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),            # 2 classes: cat / dog
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)        # flatten
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# 5. TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, correct / total


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS (plots)
# ─────────────────────────────────────────────
def save_loss_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs("artifacts", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs,   label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = "artifacts/loss_curves.png"
    plt.savefig(path)
    plt.close()
    return path


def save_confusion_matrix(labels, preds, class_names):
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    path = "artifacts/confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


# ─────────────────────────────────────────────
# 8. MAIN — ties everything together
# ─────────────────────────────────────────────
def main():
    os.makedirs("models", exist_ok=True)

    # --- Data ---
    download_data()
    prepare_data()
    train_loader, val_loader, test_loader, class_names = get_data_loaders()

    # --- Model / Loss / Optimizer ---
    model     = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # --- MLflow experiment ---
    mlflow.set_experiment("cats_dogs_classification")

    with mlflow.start_run(run_name="SimpleCNN_baseline"):

        # Log hyper-parameters
        mlflow.log_params({
            "model":         "SimpleCNN",
            "image_size":    IMAGE_SIZE,
            "batch_size":    BATCH_SIZE,
            "num_epochs":    NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer":     "Adam",
            "scheduler":     "StepLR(step=3,gamma=0.5)",
            "train_split":   TRAIN_SPLIT,
            "val_split":     VAL_SPLIT,
        })

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []

        # --- Training loop ---
        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_accs.append(tr_acc)
            val_accs.append(vl_acc)

            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{NUM_EPOCHS} | "
                  f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
                  f"Val Loss:   {vl_loss:.4f}  Acc: {vl_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Log per-epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc":  tr_acc,
                "val_loss":   vl_loss,
                "val_acc":    vl_acc,
            }, step=epoch)

        # --- Test evaluation ---
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion)
        print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds,
                                    target_names=class_names))

        # --- Save & log artifacts ---
        curves_path = save_loss_curves(train_losses, val_losses,
                                       train_accs,   val_accs)
        cm_path     = save_confusion_matrix(test_labels, test_preds, class_names)

        mlflow.log_artifact(curves_path)
        mlflow.log_artifact(cm_path)

        # --- Save model ---
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        mlflow.pytorch.log_model(model, "model")
        print(f"\nModel saved to {MODEL_SAVE_PATH}")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        print("Done! Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    main()
