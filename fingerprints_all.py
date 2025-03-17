import os
import json
import random
import yaml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm  # newly added import for progress bar
import torch
from torchvision import models, datasets


# ...existing imports for dataset handling...
from datasets import load_from_disk, Dataset, Features, Value, ClassLabel
from fin_untils import extract_fingerprint_features, SetCollator

loading = "augmented"  # Set to "augmented" or "original" based on your dataset

# ------------------------------------------------
# 1) CONFIGURATION & DEVICE SETUP
# ------------------------------------------------
# Load classifier configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
classifier_choice = config.get("classifier", "svm")  # "svm" or "rf"

# Load other configuration if needed (e.g. available models)
with open('config.json', 'r') as f:
    config = json.load(f)
models = config["models"]
selected_model = config["selected_model"]
model_name = models[selected_model]

# Set device (using MPS if available on Mac, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------
# 2) DATASET SETUP
# ------------------------------------------------
# For brevity, we assume similar dataset setup as in fingerprints_svm.py
data_dir = "dataset/onDrive-divided-cropped-augmented"
class_names = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
label2id = {name: idx for idx, name in enumerate(class_names)}
dataset_dir = "dataset/fingerprint_model_b/version_2"
train_dataset_path = os.path.join(dataset_dir, "train_dataset")
val_dataset_path   = os.path.join(dataset_dir, "val_dataset")
test_dataset_path  = os.path.join(dataset_dir, "test_dataset")
collator = SetCollator()

# Load preprocessed datasets if they exist; otherwise, build them.
if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) and os.path.exists(test_dataset_path):
    print("Loading existing datasets from disk...")
    train_dataset = load_from_disk(train_dataset_path)
    val_dataset   = load_from_disk(val_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)
elif loading == "augmented":
    # 1) Load datasets from disk
    print("Loading datasets from disk...augmented")
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

# ------------------------------------------------
# 3) CREATE FIXED-LENGTH REPRESENTATION
# ------------------------------------------------
# 1. Create a fixed-length representation using clustering
def create_fingerprint_vectors(dataset, n_clusters=64, kmeans=None):
    """
    Create fixed-length feature vectors from variable-length descriptor sets
    using KMeans clustering. Can use a pre-trained kmeans model if provided.
    """
    print(f"Creating fingerprint vectors with {n_clusters} clusters...")
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

X_train, y_train, kmeans = create_fingerprint_vectors(train_dataset, n_clusters=64)
X_val, y_val, _ = create_fingerprint_vectors(val_dataset, n_clusters=64, kmeans=kmeans)
X_test, y_test, _ = create_fingerprint_vectors(test_dataset, n_clusters=64, kmeans=kmeans)

# ------------------------------------------------
# 4) SET UP PIPELINE & GRID SEARCH BASED ON CONFIGURATION
# ------------------------------------------------
if classifier_choice.lower() == "svm":
    print("Using SVM classifier with grid search")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    param_grid = {
        'svm__C': [1, 10, 100, 1000],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'svm__kernel': ['rbf', 'poly', 'sigmoid']
    }
elif classifier_choice.lower() == "rf":
    print("Using Random Forest classifier with grid search")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10]
    }
else:
    raise ValueError("Unknown classifier choice in config.yaml. Use 'svm' or 'rf'.")

print("Starting grid search for optimal hyperparameters...")
start_time = time.time()
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    verbose=1,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Grid search completed in {time.time()-start_time:.2f} seconds")
print(f"Best parameters: {grid.best_params_}")
# NEW: Also print best grid search parameters for clarity
print("\n--- BEST GRID SEARCH PARAMETERS ---")
print(grid.best_params_)
print(f"Best CV score: {grid.best_score_:.4f}")
best_model = grid.best_estimator_

# NEW: Evaluate different cluster sizes
print("\n--- EVALUATING DIFFERENT CLUSTER SIZES ---")
cluster_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
results = []
for n_clusters in tqdm(cluster_sizes, desc="Evaluating cluster sizes"):
    print(f"Testing with {n_clusters} clusters...")
    X_train, y_train, kmeans = create_fingerprint_vectors(train_dataset, n_clusters=n_clusters)
    X_val, y_val, _ = create_fingerprint_vectors(val_dataset, n_clusters=n_clusters, kmeans=kmeans)
    X_test, y_test, _ = create_fingerprint_vectors(test_dataset, n_clusters=n_clusters, kmeans=kmeans)
    
    # Use best parameters from grid search based on classifier choice
    best_params = grid.best_params_
    
    if classifier_choice.lower() == "svm":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=best_params['svm__kernel'],
                C=best_params['svm__C'],
                gamma=best_params['svm__gamma'],
                probability=True
            ))
        ])
    elif classifier_choice.lower() == "rf":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=best_params['rf__n_estimators'],
                max_depth=best_params['rf__max_depth'],
                min_samples_split=best_params['rf__min_samples_split'],
                random_state=42
            ))
        ])
    
    pipeline.fit(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)
    test_acc = pipeline.score(X_test, y_test)
    results.append((n_clusters, val_acc, test_acc))
    print(f"  Clusters: {n_clusters}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Find best cluster size based on validation accuracy
# Get the model trained with the best cluster size
best_clusters, best_val_acc, best_test_acc = max(results, key=lambda x: x[1])
print(f"\nBest number of clusters: {best_clusters}")
print(f"  Validation accuracy: {best_val_acc:.4f}")
print(f"  Test accuracy: {best_test_acc:.4f}")

# ------------------------------------------------
# IMPORTANT: Recreate the best model with the optimal cluster size
# ------------------------------------------------
# ------------------------------------------------
# IMPORTANT: Recreate the best model with the optimal cluster size
# ------------------------------------------------
print(f"\nRecreating model with optimal cluster size ({best_clusters})...")
X_train, y_train, kmeans_best = create_fingerprint_vectors(train_dataset, n_clusters=best_clusters)
X_val, y_val, _ = create_fingerprint_vectors(val_dataset, n_clusters=best_clusters, kmeans=kmeans_best)
X_test, y_test, _ = create_fingerprint_vectors(test_dataset, n_clusters=best_clusters, kmeans=kmeans_best)

best_params = grid.best_params_
if classifier_choice.lower() == "svm":
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=best_params['svm__kernel'],
            C=best_params['svm__C'],
            gamma=best_params['svm__gamma'],
            probability=True
        ))
    ])
elif classifier_choice.lower() == "rf":
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=best_params['rf__n_estimators'],
            max_depth=best_params['rf__max_depth'],
            min_samples_split=best_params['rf__min_samples_split'],
            random_state=42
        ))
    ])

final_model.fit(X_train, y_train)

# Use this model for final evaluation instead of the original best_model
best_model = final_model

# ------------------------------------------------
# 5) EVALUATION
# ------------------------------------------------
val_pred = best_model.predict(X_val)
val_accuracy = best_model.score(X_val, y_val)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average='weighted')

test_pred = best_model.predict(X_test)
test_accuracy = best_model.score(X_test, y_test)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')

print("\nValidation Metrics:")
print(f"  Accuracy:  {val_accuracy:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1 Score:  {val_f1:.4f}")

print("\nTest Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, target_names=class_names))

plt.figure(figsize=(12,10))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix on Test Set ({classifier_choice.upper()})')
plt.tight_layout()
plt.savefig(f'confusion_matrix_{classifier_choice}.png')
print(f"Saved confusion matrix visualization to 'confusion_matrix_{classifier_choice}.png'")