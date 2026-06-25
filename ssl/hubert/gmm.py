
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.stats import mode

# =====================================================
# PATHS
# =====================================================

TRAIN_DIR = "train/mean_pooled_embeddings/layer_12"
TEST_DIR = "test/mean_pooled_embeddings/layer_12"
CSV_PATH = "test/combined_dataset_final.csv"

# =====================================================
# LABEL MAP
# =====================================================

label_map = {
    "declarative": 0,
    "interrogative": 1,
    "imperative": 2,
    "exclamatory": 3
}

id_to_label = {v: k for k, v in label_map.items()}

# =====================================================
# LOAD TRAIN (RECURSIVE)
# =====================================================

print("Loading train embeddings...")

X_train = []

for root, _, files in os.walk(TRAIN_DIR):
    for file in files:
        if file.endswith(".npy"):
            path = os.path.join(root, file)
            X_train.append(np.load(path).squeeze())

X_train = np.array(X_train)

print("Train shape:", X_train.shape)

if len(X_train) == 0:
    raise ValueError("No training embeddings found!")

# =====================================================
# INDEX TEST FILES (RECURSIVE)
# =====================================================

print("\nIndexing test embeddings...")

file_index = {}

for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.endswith(".npy"):
            file_index[file] = os.path.join(root, file)

print("Total test files indexed:", len(file_index))

# =====================================================
# LOAD CSV + TEST EMBEDDINGS
# =====================================================

print("\nLoading test embeddings + labels...")

df = pd.read_csv(CSV_PATH)

X_test = []
y_test = []

missing = []

for _, row in df.iterrows():

    file_name = row["file_name"].replace(".wav", ".npy")

    if file_name not in file_index:
        missing.append(file_name)
        continue

    X_test.append(np.load(file_index[file_name]).squeeze())
    y_test.append(label_map[row["label"]])

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Test shape:", X_test.shape)
print("Missing files:", len(missing))


# =====================================================
# PCA (OPTION 2)
# =====================================================

from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.stats import mode

# =============================
# OPTIONAL PREPROCESSING
# =============================

print("Applying Normalizer...")
scaler = Normalizer()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)

print("Applying PCA...")
pca = PCA(n_components=128, whiten=True, random_state=42)

X_train_p = pca.fit_transform(X_train_n)
X_test_p = pca.transform(X_test_n)

# =============================
# GMM MODEL
# =============================

print("Training GMM...")

gmm = GaussianMixture(
    n_components=4,
    covariance_type="full",
    random_state=42,
    n_init=5
)

gmm.fit(X_train_p)

test_clusters = gmm.predict(X_test_p)

# =============================
# CLUSTER → LABEL MAPPING
# =============================

cluster_to_label = {}

for c in range(4):

    idx = np.where(test_clusters == c)[0]

    if len(idx) == 0:
        continue

    cluster_to_label[c] = mode(y_test[idx], keepdims=True)[0][0]

# =============================
# PREDICTION
# =============================

y_pred = np.array([cluster_to_label[c] for c in test_clusters])

# =============================
# EVALUATION
# =============================

print("\nClassification Report (GMM):\n")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))