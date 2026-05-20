"""
=============================================================
train_cnn_quality.py
CNN Quality Classifier Training Script for Method B
=============================================================
This trains a small CNN to classify QR patches as PASS or REJECT.
Run this BEFORE running Dashboard_FastAPI_MethodB.py

Folder structure needed:
  qr_dataset/
    train/
      pass/    ← cropped QR patch images that are PASS
      reject/  ← cropped QR patch images that are REJECT
    val/
      pass/
      reject/

How to get training images:
  1. Run your existing Dashboard_FastAPI.py (Method A)
  2. Save the cropped QR patches when verdict = PASS or REJECT
  3. Put them in the correct folder above

Run:
    python train_cnn_quality.py

Output:
    qr_quality_cnn.pth  ← copy this to same folder as Dashboard_FastAPI_MethodB.py
=============================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================================================
# CONFIG
# =========================================================
DATASET_DIR    = "qr_dataset"       # folder with train/ and val/
CNN_INPUT_SIZE = 64                  # resize all patches to 64x64
EPOCHS         = 20
BATCH_SIZE     = 32
LR             = 0.001
SAVE_PATH      = "qr_quality_cnn.pth"


# =========================================================
# CNN MODEL  (same architecture as in Method B)
# =========================================================
class QRQualityCNN(nn.Module):
    """
    Lightweight CNN:
      Input:  (1, 64, 64) grayscale QR patch
      Output: single sigmoid score
        >= 0.5 → PASS
        <  0.5 → REJECT
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================================================
# DATA TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.ToTensor(),
])


# =========================================================
# MAIN TRAINING LOOP
# =========================================================
def train():
    # Check dataset exists
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir   = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_dir):
        print(f"ERROR: Training folder not found: {train_dir}")
        print("Please create the folder structure:")
        print("  qr_dataset/train/pass/   ← PASS QR images")
        print("  qr_dataset/train/reject/ ← REJECT QR images")
        print("  qr_dataset/val/pass/")
        print("  qr_dataset/val/reject/")
        return

    # Load datasets
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transform)

    print(f"Training samples : {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Class mapping    : {train_ds.class_to_idx}")
    print()

    # ImageFolder sorts classes alphabetically:
    #   pass   → index 0
    #   reject → index 1
    # The CNN outputs high score for class 1 (reject) by default.
    # In Dashboard_FastAPI_MethodB.py, CNN_PASS_THRESH=0.5 means:
    #   score >= 0.5 → label 1 → REJECT
    #   score <  0.5 → label 0 → PASS
    # Adjust CNN_PASS_THRESH in MethodB if your class order differs.

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = QRQualityCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ---- TRAIN ----
        model.train()
        train_loss  = 0.0
        train_correct = 0
        train_total   = 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        train_acc  = train_correct / train_total * 100
        train_loss = train_loss / train_total

        # ---- VALIDATE ----
        model.eval()
        val_correct = 0
        val_total   = 0
        val_tp = val_tn = val_fp = val_fn = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                preds   = (outputs >= 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

                # Confusion matrix counts
                # label 1 = reject (positive class)
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        val_acc = val_correct / val_total * 100
        fp_rate = val_fp / max(val_fp + val_tn, 1) * 100
        fn_rate = val_fn / max(val_fn + val_tp, 1) * 100

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  "
            f"train_acc={train_acc:.1f}%  "
            f"val_acc={val_acc:.1f}%  "
            f"FP={fp_rate:.1f}%  FN={fn_rate:.1f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> Best model saved ({val_acc:.1f}%)")

    print()
    print(f"Training complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {SAVE_PATH}")
    print()
    print("Next step: copy qr_quality_cnn.pth to the same folder")
    print("as Dashboard_FastAPI_MethodB.py and run it.")


# =========================================================
# HELPER: Save QR patches from Method A for training data
# =========================================================
def save_patch_example():
    """
    Example code showing how to save QR patches from your
    existing Dashboard_FastAPI.py to build training data.

    Add this inside InspectionProcessor.run() in Method A,
    after inspect_decoded_qr() returns a verdict:

    import uuid

    if display_verdict in ["PASS", "REJECT"] and display_pts is not None:
        x, y, w, h = cv2.boundingRect(display_pts.astype(np.int32))
        patch = enhanced_frame[max(0,y):y+h, max(0,x):x+w]
        if patch.size > 0:
            folder = f"qr_dataset/train/{display_verdict.lower()}"
            os.makedirs(folder, exist_ok=True)
            fname = f"{folder}/{uuid.uuid4().hex}.jpg"
            cv2.imwrite(fname, patch)
    """
    pass


if __name__ == "__main__":
    train()
