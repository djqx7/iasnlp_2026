import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# Checkpoint directory
# ─────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────

def load_all_embeddings(root_dir):
    X = []
    file_count = 0

    print(f"[INFO] Scanning directory: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                file_path = os.path.join(root, f)
                try:
                    emb = np.load(file_path)
                    X.append(emb)
                    file_count += 1
                except Exception as e:
                    print(f"[SKIP] {file_path} -> {e}")

    print(f"[INFO] Total .npy files found: {file_count}")

    if file_count == 0:
        raise ValueError(
            f"No .npy files found under: {root_dir}\n"
            f"Check path or file permissions."
        )

    return np.stack(X)


def load_labeled_from_csv(csv_path, embedding_root):

    df = pd.read_csv(csv_path)

    label_map = {
        "declarative": 0,
        "interrogative": 1,
        "imperative": 2,
        "exclamatory": 3
    }

    file_dict = {}
    for root, _, files in os.walk(embedding_root):
        for f in files:
            if f.endswith(".npy"):
                key = os.path.splitext(f)[0]
                file_dict[key] = os.path.join(root, f)

    print(f"[INFO] Indexed embeddings: {len(file_dict)}")

    X, y = [], []
    missing = 0

    for _, row in df.iterrows():
        key = os.path.splitext(row["file_name"])[0]
        if key in file_dict:
            emb = np.load(file_dict[key])
            X.append(emb)
            y.append(label_map[row["label"]])
        else:
            missing += 1

    print(f"[INFO] Matched samples: {len(X)}")
    print(f"[INFO] Missing samples: {missing}")

    return np.array(X), np.array(y)


def build_test_set(csv_path, embedding_root):

    df = pd.read_csv(csv_path)

    label_map = {
        "declarative": 0,
        "interrogative": 1,
        "imperative": 2,
        "exclamatory": 3
    }

    file_dict = {}
    for root, _, files in os.walk(embedding_root):
        for f in files:
            if f.endswith(".npy"):
                key = os.path.splitext(f)[0]
                file_dict[key] = os.path.join(root, f)

    X, y, valid_idx = [], [], []
    missing = 0

    for idx, row in df.iterrows():
        key = os.path.splitext(row["file_name"])[0]
        if key in file_dict:
            X.append(np.load(file_dict[key]))
            y.append(label_map[row["label"]])
            valid_idx.append(idx)
        else:
            missing += 1

    print(f"[INFO] CSV rows: {len(df)}")
    print(f"[INFO] Matched embeddings: {len(X)}")
    print(f"[INFO] Missing embeddings: {missing}")

    return np.array(X), np.array(y), np.array(valid_idx)


def check_missing_files(csv_path, embedding_root):

    df = pd.read_csv(csv_path)
    available = set()

    for root, _, files in os.walk(embedding_root):
        for f in files:
            if f.endswith(".npy"):
                available.add(os.path.splitext(f)[0])

    missing = []
    for _, row in df.iterrows():
        key = os.path.splitext(row["file_name"])[0]
        if key not in available:
            missing.append(row["file_name"])

    print(f"CSV utterances      : {len(df)}")
    print(f"Available embeddings: {len(available)}")
    print(f"Missing embeddings  : {len(missing)}")

    if len(missing) > 0:
        print("\nFirst 20 missing files:")
        for x in missing[:20]:
            print(x)

    return missing


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class LabeledDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class UnlabeledDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X).float()

    def augment(self, x):
        noise = torch.randn_like(x) * 0.01
        mask = torch.rand_like(x) > 0.1
        return x * mask + noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        return self.augment(x), self.augment(x)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def supcon_loss(features, labels, temp=0.5):
    device = features.device
    labels = labels.view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temp

    logits_mask = torch.eye(len(features), device=device)
    mask = mask * (1 - logits_mask)

    exp_logits = torch.exp(logits) * (1 - logits_mask)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    return -mean_log_prob.mean()


def simclr_loss(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp

    B = z1.shape[0]
    labels = torch.arange(B).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"[CKPT] Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    print(f"[CKPT] Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
    return epoch, val_loss


# ─────────────────────────────────────────────
# Validation loss computation
# ─────────────────────────────────────────────

def compute_val_loss(model, X_val, y_val, device, batch_size=64):
    """Compute average supervised contrastive loss on the validation set."""
    model.eval()
    loader = DataLoader(
        LabeledDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            total_loss += supcon_loss(z, y).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────

missing = check_missing_files(
    "./test/test_dataset.csv",
    "./test/mean_pooled_embeddings/layer_10"
)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

# Unlabeled train set (used for pseudo-label step)
X_train = load_all_embeddings("./train/mean_pooled_embeddings/layer_10")

# Validation set — 200 sentences with ground-truth labels
# Used during training to provide genuine supervised contrastive signal
# and to track the best checkpoint
print("\n[INFO] Loading validation set ...")
X_val, y_val = load_labeled_from_csv(
    "./val/val_dataset.csv",                    # <-- your validation CSV
    "./val/mean_pooled_embeddings/layer_10"     # <-- validation embeddings
)
print(f"[INFO] Validation set size: {len(X_val)}\n")

# Test set — kept completely separate; only touched during final evaluation
X_test, y_test, valid_idx = build_test_set(
    "./test/test_dataset.csv",
    "./test/mean_pooled_embeddings/layer_10"
)

# ─────────────────────────────────────────────
# Model + optimiser
# ─────────────────────────────────────────────

input_dim = X_train.shape[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ProjectionHead(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

best_val_loss = float("inf")
best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

for epoch in range(50):

    model.train()

    # ── 1. Supervised contrastive on validation set (ground-truth labels) ──
    # Using the 200 labelled validation sentences here gives the model a
    # clean, genuine supervised signal every epoch, which is more reliable
    # than the pseudo-labels derived from unlabelled training data alone.
    val_sup_loader = DataLoader(
        LabeledDataset(X_val, y_val),
        batch_size=64,
        shuffle=True
    )

    sup_loss_total = 0.0
    for x, y_batch in val_sup_loader:
        x, y_batch = x.to(device), y_batch.to(device)
        z = model(x)
        loss = supcon_loss(z, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sup_loss_total += loss.item()

    avg_sup_loss = sup_loss_total / len(val_sup_loader)

    # ── 2. Pseudo-label step on unlabelled train set ──
    model.eval()
    with torch.no_grad():
        Z = model(torch.tensor(X_train).float().to(device)).cpu().numpy()

    pseudo = KMeans(n_clusters=4, n_init=10).fit_predict(Z)

    pseudo_loader = DataLoader(
        LabeledDataset(X_train, pseudo),
        batch_size=128,
        shuffle=True
    )

    model.train()
    pseudo_loss_total = 0.0
    for x, y_batch in pseudo_loader:
        x, y_batch = x.to(device), y_batch.to(device)
        z = model(x)
        loss = supcon_loss(z, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pseudo_loss_total += loss.item()

    avg_pseudo_loss = pseudo_loss_total / len(pseudo_loader)

    # ── 3. Validation loss (for checkpoint selection) ──
    val_loss = compute_val_loss(model, X_val, y_val, device)

    print(
        f"Epoch {epoch:02d} | "
        f"SupCon loss (val set): {avg_sup_loss:.4f} | "
        f"Pseudo loss (train set): {avg_pseudo_loss:.4f} | "
        f"Val loss: {val_loss:.4f}"
    )

    # ── 4. Save per-epoch checkpoint ──
    epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch:02d}.pt")
    save_checkpoint(model, optimizer, epoch, val_loss, epoch_ckpt_path)

    # ── 5. Save best checkpoint ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)
        print(f"[CKPT] New best model at epoch {epoch} (val_loss={val_loss:.4f})")

print(f"\n[INFO] Training complete. Best val loss: {best_val_loss:.4f}")
print(f"[INFO] Best checkpoint saved at: {best_ckpt_path}")


# ─────────────────────────────────────────────
# Evaluation — load best checkpoint
# ─────────────────────────────────────────────

print("\n[INFO] Loading best checkpoint for evaluation ...")
load_checkpoint(model, optimizer, best_ckpt_path, device)

model.eval()
with torch.no_grad():
    Z_test = model(
        torch.tensor(X_test).float().to(device)
    ).cpu().numpy()

# Fit a linear probe on validation embeddings, evaluate on test embeddings.
# This avoids the data-leakage that would arise from fitting and evaluating
# the probe on the same (test) split.
model.eval()
with torch.no_grad():
    Z_val_probe = model(
        torch.tensor(X_val).float().to(device)
    ).cpu().numpy()

clf = LogisticRegression(max_iter=2000)
clf.fit(Z_val_probe, y_val)   # fit on validation
pred = clf.predict(Z_test)    # predict on test

# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df_test = pd.read_csv("./test/test_dataset.csv")

label_map = {
    "declarative": 0,
    "interrogative": 1,
    "imperative": 2,
    "exclamatory": 3
}

y_full = df_test["label"].map(label_map).values
y_test_final = y_full[valid_idx]

assert len(y_test_final) == len(pred), (
    f"Length mismatch: y_test={len(y_test_final)}, pred={len(pred)}"
)

accuracy  = accuracy_score(y_test_final, pred)
precision = precision_score(y_test_final, pred, average="macro")
recall    = recall_score(y_test_final, pred, average="macro")
f1        = f1_score(y_test_final, pred, average="macro")

print("\n" + "=" * 50)
print("TEST SET RESULTS (best checkpoint)")
print("=" * 50)
print(f"Accuracy         : {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro)   : {recall:.4f}")
print(f"F1-score (macro) : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test_final, pred, target_names=[
    "declarative",
    "interrogative",
    "imperative",
    "exclamatory"
]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_final, pred))