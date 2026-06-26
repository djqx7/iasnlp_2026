
    
    
    
    
    
"""
Class-wise Weighted Late Fusion — Mistral (text) + HuBERT (audio)
===================================================================
Strategy
--------
1.  Train separate classifiers on HuBERT and Mistral embeddings.
2.  Collect their soft probability outputs (logits / probabilities).
3.  Fuse with *per-class* learnable weights:
        p_fused[c] = w_audio[c] * p_audio[c] + w_text[c] * p_text[c]
    where w_audio[c] + w_text[c] = 1  (enforced via softmax).
4.  Learn the 4×2 weight matrix on the *validation* set only.
5.  Evaluate everything on the held-out *test* set.

Directory assumptions (edit the CONFIG block below):
    hubert_train/   ─ nested .npy files  (no labels needed)
    hubert_val/     ─ nested .npy files
    hubert_test/    ─ nested .npy files
    mistral_train/  ─ nested .npy files
    mistral_val/    ─ nested .npy files
    mistral_test/   ─ nested .npy files
    val.csv         ─ columns: file_name, label
    test.csv        ─ columns: file_name, label
"""

import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────
# CONFIG  ← edit these paths before running
# ─────────────────────────────────────────────────────────────
CFG = dict(
    # HuBERT embedding directories
    hubert_train_dir  = "./train/hubert_mean_pooled_embeddings/layer_9",
    hubert_val_dir    = "./val/mean_pooled_embeddings/layer_9",
    hubert_test_dir   = "./test/mean_pooled_embeddings/layer_9",

    # Mistral embedding directories
    mistral_train_dir = "./results/qwen_embeddings_train",
    mistral_val_dir   = "./results/qwen_embeddings_val",
    mistral_test_dir  = "./results/qwen_embeddings_test",

    # CSV files with ground-truth labels (columns: file_name, label)
    val_csv           = "./val/val_dataset.csv",
    test_csv          = "./test/test_dataset.csv",

    # Output
    checkpoint_dir    = "./fusion/checkpoints",

    # Training
    fusion_lr         = 1e-2,
    fusion_epochs     = 300,
    batch_size        = 64,
    device            = "cuda" if torch.cuda.is_available() else "cpu",
)

LABEL_MAP = {
    "declarative":   0,
    "interrogative": 1,
    "imperative":    2,
    "exclamatory":   3,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES  = len(LABEL_MAP)

os.makedirs(CFG["checkpoint_dir"], exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1.  Data loading helpers
# ─────────────────────────────────────────────────────────────

def _index_npy_files(root_dir: str) -> dict:
    """Walk root_dir and return {stem: full_path} for every .npy file."""
    index = {}
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                stem = os.path.splitext(f)[0]
                index[stem] = os.path.join(dirpath, f)
    return index


def load_embeddings_unlabeled(root_dir: str) -> np.ndarray:
    """Load all .npy files under root_dir into an (N, D) array."""
    index = _index_npy_files(root_dir)
    if not index:
        raise ValueError(f"No .npy files found under: {root_dir}")
    arrays = [np.load(p) for p in sorted(index.values())]
    X = np.stack(arrays)
    print(f"  [load] {root_dir}  →  {X.shape}")
    return X


def load_embeddings_labeled(root_dir: str, csv_path: str):
    """
    Match .npy files to rows in csv_path.
    CSV must have columns: file_name, label.
    Returns (X, y, valid_csv_indices).
    """
    df      = pd.read_csv(csv_path)
    index   = _index_npy_files(root_dir)

    X, y, valid_idx, missing = [], [], [], 0

    for row_idx, row in df.iterrows():
        stem = os.path.splitext(row["file_name"])[0]
        if stem in index:
            X.append(np.load(index[stem]))
            y.append(LABEL_MAP[row["label"]])
            valid_idx.append(row_idx)
        else:
            missing += 1

    print(f"  [load] {root_dir}  matched={len(X)}  missing={missing}")
    return np.array(X), np.array(y), np.array(valid_idx)


# ─────────────────────────────────────────────────────────────
# 2.  Per-modality classifiers  (Logistic Regression probes)
# ─────────────────────────────────────────────────────────────

def train_probe(X_train: np.ndarray, y_train: np.ndarray,
                name: str, cfg: dict) -> LogisticRegression:
    """
    Fit a logistic-regression probe on the given embeddings and save it
    as  checkpoints/probe_<name>.pt  so it can be reloaded independently.
    """
    print(f"\n[probe] Training {name} probe  …")
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             multi_class="multinomial")
    clf.fit(X_train, y_train)
    print(f"[probe] {name} probe ready.")
    save_probe(clf, name, cfg)
    return clf


def get_probs(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return (N, C) probability matrix."""
    return clf.predict_proba(X)          # shape (N, num_classes)


def save_probe(clf: LogisticRegression, name: str, cfg: dict) -> str:
    """
    Persist a fitted sklearn probe as a .pt file via torch.save.
    Returns the saved path.
    """
    path = os.path.join(cfg["checkpoint_dir"], f"probe_{name}.pt")
    torch.save({"probe": clf, "name": name}, path)
    print(f"[CKPT] Probe '{name}' saved  →  {path}")
    return path


def load_probe(name: str, cfg: dict) -> LogisticRegression:
    """Load a previously saved sklearn probe from a .pt file."""
    path = os.path.join(cfg["checkpoint_dir"], f"probe_{name}.pt")
    data = torch.load(path, map_location="cpu")
    print(f"[CKPT] Probe '{name}' loaded  ←  {path}")
    return data["probe"]


# ─────────────────────────────────────────────────────────────
# 3.  Class-wise weighted late-fusion module
# ─────────────────────────────────────────────────────────────

class ClasswiseFusion(nn.Module):
    """
    Learnable per-class fusion weights for two modalities.

    Raw logit w[c, m] → softmax over modality axis  →  weight in [0,1].
    This enforces w_audio[c] + w_text[c] = 1 for every class c.

    Shape of internal parameter: (num_classes, num_modalities=2)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Initialise to equal weights (0.5 / 0.5) for every class
        self.logits = nn.Parameter(torch.zeros(num_classes, 2))

    def weights(self) -> torch.Tensor:
        """Return normalised weights  (num_classes, 2)."""
        return torch.softmax(self.logits, dim=1)

    def forward(self, p_audio: torch.Tensor,
                p_text: torch.Tensor) -> torch.Tensor:
        """
        p_audio : (N, C)
        p_text  : (N, C)
        returns   (N, C)  fused probability distribution
        """
        w = self.weights()                      # (C, 2)
        w_audio = w[:, 0]                       # (C,)
        w_text  = w[:, 1]                       # (C,)

        # Element-wise weighting then sum across modalities
        fused = p_audio * w_audio + p_text * w_text   # (N, C)
        return fused


# ─────────────────────────────────────────────────────────────
# 4.  Fusion dataset
# ─────────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    def __init__(self, p_audio: np.ndarray, p_text: np.ndarray,
                 y: np.ndarray):
        self.p_audio = torch.tensor(p_audio, dtype=torch.float32)
        self.p_text  = torch.tensor(p_text,  dtype=torch.float32)
        self.y       = torch.tensor(y,        dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.p_audio[i], self.p_text[i], self.y[i]


# ─────────────────────────────────────────────────────────────
# 5.  Train fusion weights on validation set
# ─────────────────────────────────────────────────────────────

def save_fusion_checkpoint(fusion: "ClasswiseFusion",
                           optimizer: optim.Optimizer,
                           epoch: int, loss: float, path: str) -> None:
    """Save a full fusion training snapshot to a .pt file."""
    torch.save({
        "epoch":                epoch,
        "fusion_state_dict":    fusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             loss,
    }, path)
    print(f"[CKPT] Fusion epoch {epoch:03d}  loss={loss:.4f}  →  {path}")


def load_fusion_checkpoint(path: str, cfg: dict,
                           ) -> tuple["ClasswiseFusion", optim.Optimizer, int, float]:
    """
    Restore a fusion model + optimiser from a .pt checkpoint.
    Returns (fusion, optimizer, epoch, val_loss).
    """
    device    = cfg["device"]
    ckpt      = torch.load(path, map_location=device)
    fusion    = ClasswiseFusion(NUM_CLASSES).to(device)
    optimizer = optim.Adam(fusion.parameters(), lr=cfg["fusion_lr"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch     = ckpt["epoch"]
    val_loss  = ckpt["val_loss"]
    print(f"[CKPT] Fusion checkpoint loaded  ←  {path}  "
          f"(epoch={epoch}, val_loss={val_loss:.4f})")
    return fusion, optimizer, epoch, val_loss


def train_fusion(p_audio_val: np.ndarray, p_text_val: np.ndarray,
                 y_val: np.ndarray, cfg: dict) -> ClasswiseFusion:
    """
    Optimise the 4×2 class-wise weight matrix on the validation set.
    Only 8 scalar parameters are learned — very low risk of overfitting
    even with 200 validation samples.

    Checkpoints saved
    -----------------
    • checkpoints/fusion_epoch_NNN.pt  — one per epoch (full snapshot)
    • checkpoints/fusion_best.pt       — overwritten whenever val loss improves
    """
    device    = cfg["device"]
    fusion    = ClasswiseFusion(NUM_CLASSES).to(device)
    optimizer = optim.Adam(fusion.parameters(), lr=cfg["fusion_lr"])
    criterion = nn.CrossEntropyLoss()

    dataset = FusionDataset(p_audio_val, p_text_val, y_val)
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    best_ckpt_path = os.path.join(cfg["checkpoint_dir"], "fusion_best.pt")

    print("\n[fusion] Training class-wise fusion weights on validation set …")
    best_loss, best_state = float("inf"), None

    for epoch in range(cfg["fusion_epochs"]):
        fusion.train()
        epoch_loss = 0.0

        for p_a, p_t, y_batch in loader:
            p_a, p_t, y_batch = p_a.to(device), p_t.to(device), y_batch.to(device)

            fused = fusion(p_a, p_t)            # (N, C)
            # Use log of fused probs as logits for cross-entropy
            log_fused = torch.log(fused + 1e-9)
            loss = criterion(log_fused, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # ── Per-epoch checkpoint ──────────────────────────────
        epoch_ckpt_path = os.path.join(
            cfg["checkpoint_dir"], f"fusion_epoch_{epoch:03d}.pt")
        save_fusion_checkpoint(fusion, optimizer, epoch, avg_loss, epoch_ckpt_path)

        # ── Best-model checkpoint ─────────────────────────────
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in fusion.state_dict().items()}
            save_fusion_checkpoint(fusion, optimizer, epoch,
                                   avg_loss, best_ckpt_path)
            print(f"[CKPT] ★ New best fusion model  (loss={avg_loss:.4f})")

        if (epoch + 1) % 50 == 0:
            w = fusion.weights().detach().cpu().numpy()
            print(f"  epoch {epoch+1:>4d}  loss={avg_loss:.4f}  "
                  f"weights(audio|text)= {np.round(w, 3).tolist()}")

    # Restore best weights before returning
    fusion.load_state_dict(best_state)
    print(f"\n[fusion] Training complete. Best val loss: {best_loss:.4f}")
    print(f"[fusion] Best checkpoint  →  {best_ckpt_path}")

    return fusion


# ─────────────────────────────────────────────────────────────
# 6.  Evaluation helper
# ─────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, split: str = "Test"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    print(f"\n{'='*55}")
    print(f"  {split} Results")
    print(f"{'='*55}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro)   : {rec:.4f}")
    print(f"  F1-score (macro) : {f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=label_names,
                                zero_division=0))
    print(f"  Confusion Matrix (rows=true, cols=pred):\n")
    cm = confusion_matrix(y_true, y_pred)
    # Pretty-print with labels
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df.to_string())
    print()

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)


# ─────────────────────────────────────────────────────────────
# 7.  Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    """
    Corrected pipeline
    ──────────────────
    The original approach used KMeans pseudo-labels to train the probes,
    which caused the probes to output arbitrarily-numbered cluster IDs
    instead of real class probabilities — making fusion impossible.

    Correct approach
    ────────────────
    Step 1  Train probes on the VALIDATION set with GROUND-TRUTH labels.
            200 labelled samples is more than enough for logistic regression.

    Step 2  Obtain out-of-fold probe probabilities on the validation set
            using 5-fold cross-validation so that the fusion learner never
            trains on the same samples the probes were fitted on.

    Step 3  Train the 4×2 fusion weight matrix on those out-of-fold probs.

    Step 4  Re-fit final probes on ALL 200 val samples (full fit).

    Step 5  Get test-set probabilities from the final probes and apply
            the learned fusion weights → evaluate on test.
    """
    from sklearn.model_selection import StratifiedKFold

    device = CFG["device"]
    print(f"\n[INFO] Device: {device}")

    # ── 7.1  Load labeled val & test embeddings ───────────────
    print("\n[INFO] Loading HuBERT val & test embeddings …")
    X_hubert_val,  y_val,  _ = load_embeddings_labeled(
        CFG["hubert_val_dir"],  CFG["val_csv"])
    X_hubert_test, y_test, _ = load_embeddings_labeled(
        CFG["hubert_test_dir"], CFG["test_csv"])

    print("\n[INFO] Loading Mistral val & test embeddings …")
    X_mistral_val,  _, _ = load_embeddings_labeled(
        CFG["mistral_val_dir"],  CFG["val_csv"])
    X_mistral_test, _, _ = load_embeddings_labeled(
        CFG["mistral_test_dir"], CFG["test_csv"])

    assert len(X_hubert_val)  == len(X_mistral_val),  "Val size mismatch!"
    assert len(X_hubert_test) == len(X_mistral_test), "Test size mismatch!"
    print(f"\n[INFO] Val size : {len(X_hubert_val)}")
    print(f"[INFO] Test size: {len(X_hubert_test)}")

    # ── 7.2  Out-of-fold probabilities on validation set ─────
    # We use 5-fold CV so that every val sample gets a probability
    # estimate from a probe that was NOT trained on that sample.
    # This prevents the fusion from exploiting probe over-fitting.
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_hubert  = np.zeros((len(y_val),  NUM_CLASSES))
    oof_mistral = np.zeros((len(y_val), NUM_CLASSES))

    print(f"\n[INFO] Generating out-of-fold probabilities ({N_FOLDS}-fold CV) …")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_hubert_val, y_val)):
        # HuBERT fold
        clf_h = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                   multi_class="multinomial")
        clf_h.fit(X_hubert_val[tr_idx], y_val[tr_idx])
        oof_hubert[va_idx] = clf_h.predict_proba(X_hubert_val[va_idx])

        # Mistral fold
        clf_m = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                   multi_class="multinomial")
        clf_m.fit(X_mistral_val[tr_idx], y_val[tr_idx])
        oof_mistral[va_idx] = clf_m.predict_proba(X_mistral_val[va_idx])

        oof_acc_h = accuracy_score(y_val[va_idx], np.argmax(oof_hubert[va_idx], 1))
        oof_acc_m = accuracy_score(y_val[va_idx], np.argmax(oof_mistral[va_idx], 1))
        print(f"  Fold {fold+1}/{N_FOLDS}  HuBERT acc={oof_acc_h:.3f}  "
              f"Mistral acc={oof_acc_m:.3f}")

    # Sanity-check OOF accuracy — should be close to your reported 62 / 64 %
    print(f"\n[INFO] OOF HuBERT  accuracy : "
          f"{accuracy_score(y_val, np.argmax(oof_hubert,  1)):.4f}")
    print(f"[INFO] OOF Mistral accuracy : "
          f"{accuracy_score(y_val, np.argmax(oof_mistral, 1)):.4f}")

    # ── 7.3  Train fusion weights on OOF probabilities ───────
    fusion = train_fusion(oof_hubert, oof_mistral, y_val, CFG)

    # Print learned per-class weights
    w = fusion.weights().detach().cpu().numpy()
    print("\n[INFO] Learned class-wise fusion weights:")
    print(f"  {'Class':<16}  {'w_HuBERT':>10}  {'w_Mistral':>10}")
    print(f"  {'-'*40}")
    for c in range(NUM_CLASSES):
        print(f"  {IDX_TO_LABEL[c]:<16}  {w[c,0]:>10.4f}  {w[c,1]:>10.4f}")

    # ── 7.4  Final probes — fit on ALL 200 val samples ───────
    print("\n[INFO] Fitting final probes on full validation set …")
    clf_hubert  = train_probe(X_hubert_val,  y_val, "hubert",  CFG)
    clf_mistral = train_probe(X_mistral_val, y_val, "mistral", CFG)

    # ── 7.5  Test-set probabilities ───────────────────────────
    p_hubert_test  = get_probs(clf_hubert,  X_hubert_test)
    p_mistral_test = get_probs(clf_mistral, X_mistral_test)

    # ── 7.6  Fuse and evaluate on TEST set ───────────────────
    fusion.eval()
    with torch.no_grad():
        fused_test = fusion(
            torch.tensor(p_hubert_test,  dtype=torch.float32).to(device),
            torch.tensor(p_mistral_test, dtype=torch.float32).to(device),
        ).cpu().numpy()

    pred_test = np.argmax(fused_test, axis=1)
    metrics   = evaluate(y_test, pred_test, split="Fused (test) ← FINAL")

    # ── 7.7  Save results ────────────────────────────────────
    results_path = os.path.join(CFG["checkpoint_dir"], "fusion_results.csv")
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    print(f"[INFO] Results saved to {results_path}")

    return metrics


if __name__ == "__main__":
    main()
