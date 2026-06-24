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

    # build index: REMOVE EXTENSION (.npy)
    file_dict = {}

    for root, _, files in os.walk(embedding_root):
        for f in files:
            if f.endswith(".npy"):
                key = os.path.splitext(f)[0]   # <-- IMPORTANT FIX
                file_dict[key] = os.path.join(root, f)

    print(f"[INFO] Indexed embeddings: {len(file_dict)}")

    X, y = [], []
    missing = 0

    for _, row in df.iterrows():

        # remove .wav extension from CSV file name
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

    X, y = [], []

    missing = 0

    for _, row in df.iterrows():
        key = os.path.splitext(row["file_name"])[0]

        if key in file_dict:
            X.append(np.load(file_dict[key]))
            y.append(label_map[row["label"]])
        else:
            missing += 1

    print(f"[INFO] Missing samples: {missing}")
    print(f"[INFO] Final test size: {len(X)}")

    return np.array(X), np.array(y)

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
    
X_train = load_all_embeddings("./train/mean_pooled_embeddings/")

X_test, y_test = build_test_set(
    "./test/combined_dataset_final.csv",
    "./test/mean_pooled_embeddings"
)

input_dim = X_train.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ProjectionHead(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):

    model.train()

    # -------------------
    # supervised contrastive
    # -------------------
    labeled_loader = DataLoader(
        LabeledDataset(X_test, y_test),
        batch_size=64,
        shuffle=True
    )

    for x, y in labeled_loader:
        x, y = x.to(device), y.to(device)
        z = model(x)

        loss = supcon_loss(z, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -------------------
    # pseudo-label step
    # -------------------
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
    for x, y in pseudo_loader:
        x, y = x.to(device), y.to(device)
        z = model(x)

        loss = supcon_loss(z, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")
    
model.eval()

with torch.no_grad():
    Z_test = model(torch.tensor(X_test).float().to(device)).cpu().numpy()
    
clf = LogisticRegression(max_iter=2000)
clf.fit(Z_t, y_labeled)

pred = clf.predict(Z_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("./test/combined_dataset_final.csv")

label_map = {
    "declarative": 0,
    "interrogative": 1,
    "imperative": 2,
    "exclamatory": 3
}

y_full = df["label"].map(label_map).values

X_test, valid_idx = build_test_set(
    "./test/combined_dataset_final.csv",
    "./train/mean_pooled_embeddings"
)

y_test = y_full[valid_idx]

assert len(y_test) == len(pred)

accuracy = accuracy_score(y_test, pred)

precision = precision_score(y_test, pred, average="macro")
recall = recall_score(y_test, pred, average="macro")
f1 = f1_score(y_test, pred, average="macro")

print("Accuracy:", accuracy)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)

print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=[
    "declarative",
    "interrogative",
    "imperative",
    "exclamatory"
]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))
