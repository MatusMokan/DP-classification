from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
from torchvision import models, datasets


from fin_untils import extract_fingerprint_features
from fin_untils import SetCollator

from datasets import load_dataset, Dataset, Features, Value, ClassLabel
from datasets import load_from_disk

loading = "augmented"  # Set to "augmented" or "original" based on your needs

# ------------------------------------------------
# 1) CONFIGURATION & DEVICE SETUP
# ------------------------------------------------
# Load configuration from config.json (e.g. available model names)
with open('config.json', 'r') as f:
    config = json.load(f)
models = config["models"]
selected_model = config["selected_model"]
model_name = models[selected_model]

# Set device (using MPS if available on Mac, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------
# 2) DATASET SETUP: READ CLASSES & DEFINE PATHS
# ------------------------------------------------
# Data directory with augmented retinal images.
data_dir = "dataset/onDrive-divided-cropped-augmented"
# Each subfolder in data_dir is assumed to be one class (person).
class_names = sorted(
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
)
# Create mappings between class names and numeric labels.
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}
print(f"Found {len(class_names)} classes: {class_names}")

# Paths for saving preprocessed Hugging Face datasets.
dataset_dir = "dataset/fingerprint_model_b/version_2"
train_dataset_path = os.path.join(dataset_dir, "train_dataset")
val_dataset_path   = os.path.join(dataset_dir, "val_dataset")
test_dataset_path  = os.path.join(dataset_dir, "test_dataset")

# Use the custom collator for set-based inputs.
collator = SetCollator()

# Load preprocessed datasets if they exist; otherwise, build them.
if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) and os.path.exists(test_dataset_path):
    print("Loading existing datasets from disk...")
    train_dataset = load_from_disk(train_dataset_path)
    val_dataset   = load_from_disk(val_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)
if loading == "augmented":
    # 1) Load datasets from disk
    data_root = "./dataset/augmented_13"
    train_dir = os.path.join(data_root, "train_data")
    val_dir = os.path.join(data_root, "val_data")
    test_dir = os.path.join(data_root, "test_data")

    # Load as PyTorch ImageFolder datasets
    train_folder = datasets.ImageFolder(root=train_dir)
    val_folder = datasets.ImageFolder(root=val_dir)
    test_folder = datasets.ImageFolder(root=test_dir)

    print(f"Train dataset size: {len(train_folder)}")
    print(f"Val dataset size: {len(val_folder)}")
    print(f"Test dataset size: {len(test_folder)}")

    # Convert to Hugging Face datasets
    train_files = [img_path for img_path, _ in train_folder.samples]
    train_labels = [label for _, label in train_folder.samples]
    val_files = [img_path for img_path, _ in val_folder.samples]
    val_labels = [label for _, label in val_folder.samples]
    test_files = [img_path for img_path, _ in test_folder.samples]
    test_labels = [label for _, label in test_folder.samples]

    # Create Hugging Face datasets
    train_dict = {"path": train_files, "label": train_labels}
    val_dict = {"path": val_files, "label": val_labels}
    test_dict = {"path": test_files, "label": test_labels}
    
    features = Features({
        "path": Value("string"),
        "label": ClassLabel(num_classes=len(class_names), names=class_names),
    })
    
    train_dataset = Dataset.from_dict(train_dict).cast(features)
    val_dataset = Dataset.from_dict(val_dict).cast(features)
    test_dataset = Dataset.from_dict(test_dict).cast(features)
    
    # Now we can use the map method on Hugging Face datasets
    train_dataset = train_dataset.map(extract_fingerprint_features)
    val_dataset = val_dataset.map(extract_fingerprint_features)
    test_dataset = test_dataset.map(extract_fingerprint_features)
    
    # Remove the file path column
    train_dataset = train_dataset.remove_columns(["path"])
    val_dataset = val_dataset.remove_columns(["path"])
    test_dataset = test_dataset.remove_columns(["path"])
    
    # Save the datasets
    train_dataset.save_to_disk(train_dataset_path)
    val_dataset.save_to_disk(val_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)
else:
    print("Building new datasets from", data_dir)
    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    random.seed(42)
    # Example splitting: 17 train, 4 validation, 4 test images per class.
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
        images = [os.path.join(class_dir, f) for f in images]
        random.shuffle(images)
        train_imgs = images[:17]
        val_imgs   = images[17:21]
        test_imgs  = images[21:25]
        label = label2id[class_name]
        train_files.extend(train_imgs)
        val_files.extend(val_imgs)
        test_files.extend(test_imgs)
        train_labels.extend([label] * len(train_imgs))
        val_labels.extend([label] * len(val_imgs))
        test_labels.extend([label] * len(test_imgs))
    os.makedirs(dataset_dir, exist_ok=True)
    train_dict = {"path": train_files, "label": train_labels}
    val_dict   = {"path": val_files,   "label": val_labels}
    test_dict  = {"path": test_files,  "label": test_labels}
    features = Features({
        "path": Value("string"),
        "label": ClassLabel(num_classes=len(class_names), names=class_names),
    })
    train_dataset = Dataset.from_dict(train_dict).cast(features)
    val_dataset   = Dataset.from_dict(val_dict).cast(features)
    test_dataset  = Dataset.from_dict(test_dict).cast(features)
    # Map the feature extraction function onto each dataset.
    train_dataset = train_dataset.map(extract_fingerprint_features)
    val_dataset   = val_dataset.map(extract_fingerprint_features)
    test_dataset  = test_dataset.map(extract_fingerprint_features)
    # Remove the file path column (no longer needed).
    train_dataset = train_dataset.remove_columns(["path"])
    val_dataset   = val_dataset.remove_columns(["path"])
    test_dataset  = test_dataset.remove_columns(["path"])
    # Save the datasets for later reuse.
    train_dataset.save_to_disk(train_dataset_path)
    val_dataset.save_to_disk(val_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)


# 1. Create a fixed-length representation using clustering
def create_fingerprint_vectors(dataset, n_clusters=64, kmeans=None):
    """
    Create fixed-length feature vectors from variable-length descriptor sets
    using KMeans clustering. Can use a pre-trained kmeans model if provided.
    """
    # If kmeans is not provided, train it on this dataset's descriptors
    if kmeans is None:
        # Collect all descriptors from the training set
        all_descriptors = []
        for sample in dataset:
            all_descriptors.extend(sample["fingerprint_features"])
        
        # Train KMeans to create a "vocabulary" of descriptor patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(all_descriptors)
    
    # Create histogram features for each fingerprint
    feature_vectors = []
    labels = []
    for sample in dataset:
        if not sample["fingerprint_features"]:
            # Handle empty descriptor sets
            histogram = np.zeros(n_clusters)
        else:
            # Assign each descriptor to a cluster
            cluster_ids = kmeans.predict(sample["fingerprint_features"])
            # Create histogram
            histogram = np.bincount(cluster_ids, minlength=n_clusters)
            # Normalize
            histogram = histogram.astype(float) / len(sample["fingerprint_features"])
        
        feature_vectors.append(histogram)
        labels.append(sample["label"])
    
    return np.array(feature_vectors), np.array(labels), kmeans

# 2. Train and evaluate SVM
# Create feature vectors
# Train on training data and get the kmeans model
X_train, y_train, kmeans = create_fingerprint_vectors(train_dataset, n_clusters=64)

# Reuse the same kmeans model for validation and test data
X_val, y_val, _ = create_fingerprint_vectors(val_dataset, n_clusters=64, kmeans=kmeans)
X_test, y_test, _ = create_fingerprint_vectors(test_dataset, n_clusters=64, kmeans=kmeans)

# Train SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
])

pipeline.fit(X_train, y_train)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------
# PART 1: HYPERPARAMETER TUNING WITH GRID SEARCH
# ------------------------------------------------
print("Starting grid search for optimal hyperparameters...")
start_time = time.time()

# Define parameter grid
param_grid = {
    'svm__C': [1, 10, 100, 1000],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svm__kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid search with cross-validation
grid = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    verbose=1, 
    scoring='accuracy',
    n_jobs=-1  # Use all available cores
)

grid.fit(X_train, y_train)
print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds")
print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-validation score: {grid.best_score_:.4f}")

# Get the best model
best_model = grid.best_estimator_

# ------------------------------------------------
# PART 1.3: XGBOOST CLASSIFIER
# ------------------------------------------------
print("\n--- XGBOOST CLASSIFIER EVALUATION ---")
start_time = time.time()

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
print("Training XGBoost classifier...")
xgb.fit(X_train, y_train)

# Evaluate XGBoost
xgb_val_acc = xgb.score(X_val, y_val)
xgb_test_acc = xgb.score(X_test, y_test)
xgb_val_pred = xgb.predict(X_val)
xgb_test_pred = xgb.predict(X_test)

xgb_val_precision, xgb_val_recall, xgb_val_f1, _ = precision_recall_fscore_support(
    y_val, xgb_val_pred, average='weighted')
xgb_test_precision, xgb_test_recall, xgb_test_f1, _ = precision_recall_fscore_support(
    y_test, xgb_test_pred, average='weighted')

print(f"XGBoost training completed in {time.time() - start_time:.2f} seconds")

print("\nXGBoost Validation Set Metrics:")
print(f"  Accuracy:  {xgb_val_acc:.4f}")
print(f"  Precision: {xgb_val_precision:.4f}")
print(f"  Recall:    {xgb_val_recall:.4f}")
print(f"  F1 Score:  {xgb_val_f1:.4f}")

print("\nXGBoost Test Set Metrics:")
print(f"  Accuracy:  {xgb_test_acc:.4f}")
print(f"  Precision: {xgb_test_precision:.4f}")
print(f"  Recall:    {xgb_test_recall:.4f}")
print(f"  F1 Score:  {xgb_test_f1:.4f}")

# Plot XGBoost confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, xgb_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('XGBoost Model Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_xgboost.png')
print("Saved XGBoost confusion matrix visualization to 'confusion_matrix_xgboost.png'")

# ------------------------------------------------
# PART 1.4: STACKING CLASSIFIER
# ------------------------------------------------
print("\n--- STACKING CLASSIFIER EVALUATION ---")
start_time = time.time()

# Create base estimators for stacking
base_estimators = [
    ('svm_rbf', SVC(kernel='rbf', C=10, gamma='scale', probability=True)),
    ('svm_poly', SVC(kernel='poly', C=100, degree=2, probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42))
]

# Create and train stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)

print("Training Stacking classifier...")
stacking_clf.fit(X_train, y_train)

# Evaluate Stacking classifier
stacking_val_acc = stacking_clf.score(X_val, y_val)
stacking_test_acc = stacking_clf.score(X_test, y_test)
stacking_val_pred = stacking_clf.predict(X_val)
stacking_test_pred = stacking_clf.predict(X_test)

stacking_val_precision, stacking_val_recall, stacking_val_f1, _ = precision_recall_fscore_support(
    y_val, stacking_val_pred, average='weighted')
stacking_test_precision, stacking_test_recall, stacking_test_f1, _ = precision_recall_fscore_support(
    y_test, stacking_test_pred, average='weighted')

print(f"Stacking classifier training completed in {time.time() - start_time:.2f} seconds")

print("\nStacking Classifier Validation Set Metrics:")
print(f"  Accuracy:  {stacking_val_acc:.4f}")
print(f"  Precision: {stacking_val_precision:.4f}")
print(f"  Recall:    {stacking_val_recall:.4f}")
print(f"  F1 Score:  {stacking_val_f1:.4f}")

print("\nStacking Classifier Test Set Metrics:")
print(f"  Accuracy:  {stacking_test_acc:.4f}")
print(f"  Precision: {stacking_test_precision:.4f}")
print(f"  Recall:    {stacking_test_recall:.4f}")
print(f"  F1 Score:  {stacking_test_f1:.4f}")

# Plot stacking classifier confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, stacking_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Stacking Classifier Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_stacking.png')
print("Saved Stacking classifier confusion matrix visualization to 'confusion_matrix_stacking.png'")

# ------------------------------------------------
# PART 1.5: ENSEMBLE CLASSIFIER
# ------------------------------------------------
print("\n--- ENSEMBLE CLASSIFIER EVALUATION ---")
start_time = time.time()

# Create multiple classifiers with different parameters
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_poly = SVC(kernel='poly', C=100, gamma='scale', probability=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

# Combine them in a voting classifier
ensemble = VotingClassifier(
    estimators=[('rbf', svm_rbf), ('poly', svm_poly), ('rf', rf)],
    voting='soft'
)

# Train the ensemble
print("Training ensemble classifier...")
ensemble.fit(X_train, y_train)

# Evaluate the ensemble
ensemble_val_acc = ensemble.score(X_val, y_val)
ensemble_test_acc = ensemble.score(X_test, y_test)
ensemble_val_pred = ensemble.predict(X_val)
ensemble_test_pred = ensemble.predict(X_test)

val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    y_val, ensemble_val_pred, average='weighted')
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    y_test, ensemble_test_pred, average='weighted')

print(f"Ensemble training completed in {time.time() - start_time:.2f} seconds")

print("\nEnsemble Validation Set Metrics:")
print(f"  Accuracy:  {ensemble_val_acc:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1 Score:  {val_f1:.4f}")

print("\nEnsemble Test Set Metrics:")
print(f"  Accuracy:  {ensemble_test_acc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# Plot ensemble confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, ensemble_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Ensemble Model Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_ensemble.png')
print("Saved ensemble confusion matrix visualization to 'confusion_matrix_ensemble.png'")

# ------------------------------------------------
# PART 2: COMPREHENSIVE EVALUATION
# ------------------------------------------------
print("\n--- EVALUATION WITH BEST MODEL ---")

# Evaluate on validation set
val_pred = best_model.predict(X_val)
val_accuracy = best_model.score(X_val, y_val)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average='weighted')

# Evaluate on test set
test_pred = best_model.predict(X_test)
test_accuracy = best_model.score(X_test, y_test)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')

# Print results
print("\nValidation Set Metrics:")
print(f"  Accuracy:  {val_accuracy:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1 Score:  {val_f1:.4f}")

print("\nTest Set Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# ------------------------------------------------
# PART 3: VISUALIZE RESULTS
# ------------------------------------------------
print("\n--- DETAILED CLASSIFICATION REPORT ---")
print("\nTest Set Classification Report:")
print(classification_report(y_test, test_pred, target_names=class_names))

# Plot confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix_svm.png')
print("Saved confusion matrix visualization to 'confusion_matrix_svm.png'")

# Class-wise accuracy
class_correct = {}
class_total = {}
for true, pred in zip(y_test, test_pred):
    class_name = class_names[true]
    if class_name not in class_total:
        class_total[class_name] = 0
        class_correct[class_name] = 0
    class_total[class_name] += 1
    if true == pred:
        class_correct[class_name] += 1

# Plot per-class accuracy
plt.figure(figsize=(12, 6))
accuracies = [class_correct.get(c, 0) / class_total.get(c, 1) for c in class_names]
sns.barplot(x=class_names, y=accuracies)
plt.title('Per-class Accuracy')
plt.xticks(rotation=90)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('per_class_accuracy_svm.png')
print("Saved per-class accuracy visualization to 'per_class_accuracy_svm.png'")

# Try different cluster sizes
print("\n--- EVALUATING DIFFERENT CLUSTER SIZES ---")
cluster_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
results = []

for n_clusters in tqdm(cluster_sizes, desc="Evaluating cluster sizes"):
    print(f"Testing with {n_clusters} clusters...")
    X_train, y_train, kmeans = create_fingerprint_vectors(train_dataset, n_clusters=n_clusters)
    X_val, y_val, _ = create_fingerprint_vectors(val_dataset, n_clusters=n_clusters, kmeans=kmeans)
    X_test, y_test, _ = create_fingerprint_vectors(test_dataset, n_clusters=n_clusters, kmeans=kmeans)
    
    # Use best parameters from grid search
    best_params = grid.best_params_
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=best_params['svm__kernel'],
            C=best_params['svm__C'],
            gamma=best_params['svm__gamma']
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)
    test_acc = pipeline.score(X_test, y_test)
    results.append((n_clusters, val_acc, test_acc))
    print(f"  Clusters: {n_clusters}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Find best cluster size
best_clusters, best_val_acc, best_test_acc = max(results, key=lambda x: x[1])
print(f"\nBest number of clusters: {best_clusters}")
print(f"  Validation accuracy: {best_val_acc:.4f}")
print(f"  Test accuracy: {best_test_acc:.4f}")