import os
import random
import json
import torch
import numpy as np
from datasets import Dataset, Features, ClassLabel, Value, load_from_disk
import fingerprint_feature_extractor  # Your library for minutiae extraction
import cv2
import math
import torch.nn.functional as F
from typing import Any, Dict, List
from dataclasses import dataclass

# ------------------------------
# 1) CONFIGURATION & DEVICE SETUP
# ------------------------------
# Load configuration from a JSON file. This file should define model names and which one to use.
with open('config.json', 'r') as f:
    config = json.load(f)
models = config["models"]
selected_model = config["selected_model"]
model_name = models[selected_model]

# Set up the device (use MPS on macOS if available, otherwise fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 2) DATA DIRECTORY & CLASS NAMES
# ------------------------------
# Specify the directory where your augmented images are stored.
data_dir = "dataset/onDrive-divided-cropped-augmented"

# Get a sorted list of class names (each subfolder represents a class)
class_names = sorted(
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
)

# Create a mapping from class names to numeric labels and vice versa.
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}
print(f"Found {len(class_names)} classes: {class_names}")

# Set paths for saving the Hugging Face datasets (train, validation, test)
dataset_dir = "dataset/fingerprint"
train_dataset_path = os.path.join(dataset_dir, "train_dataset")
val_dataset_path   = os.path.join(dataset_dir, "val_dataset")
test_dataset_path  = os.path.join(dataset_dir, "test_dataset")

# ------------------------------
# 3) MINUTIAE FEATURE HELPERS
# ------------------------------

def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_degrees(x1, y1, x2, y2):
    """
    Calculate the angle (in degrees) between the line joining (x1, y1) and (x2, y2).
    The result is in the range [-180, 180].
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def compute_4NN_descriptor(minutia, all_minutiae, k_neighbors=4):
    """
    For a given minutia, find its 4 nearest neighbors from all_minutiae.
    Compute an 8-dimensional descriptor containing:
        [d1, d2, d3, d4, angle1, angle2, angle3, angle4]
    If there are fewer than 4 neighbors, pad with zeros.
    """
    cx, cy = minutia.locX, minutia.locY
    dist_list = []
    # Compute distances from the current minutia to every other minutia
    for m in all_minutiae:
        if m is minutia:
            continue
        dist = euclidean_distance(cx, cy, m.locX, m.locY)
        dist_list.append((dist, m))
    # Sort the distances in ascending order and select the 4 nearest
    dist_list.sort(key=lambda x: x[0])
    four_nearest = dist_list[:k_neighbors]
    distances = []
    angles = []
    # For each nearest minutia, compute the distance and the relative angle
    for dist, neigh in four_nearest:
        nx, ny = neigh.locX, neigh.locY
        ang = angle_degrees(cx, cy, nx, ny)
        distances.append(dist)
        angles.append(ang)
    # Pad with zeros if fewer than 4 neighbors are found
    while len(distances) < 4:
        distances.append(0.0)
        angles.append(0.0)
    return distances + angles  # Return the concatenated 8D descriptor

def build_retina_descriptor(terminations, bifurcations):
    """
    Instead of averaging all descriptors into a single vector (which loses detail),
    this function returns the full list of 8D descriptors extracted from the minutiae points.
    If no minutiae are detected, returns an empty list.
    """
    all_min = terminations + bifurcations
    if len(all_min) == 0:
        return []  # Return an empty list if no minutiae are found
    descriptors = []
    for m in all_min:
        desc = compute_4NN_descriptor(m, all_min, k_neighbors=4)
        descriptors.append(desc)
    return descriptors  # List of 8D descriptors for the image

def extract_fingerprint_features(example):
    """
    For a given example (which includes the path to an image),
    read the image, extract minutiae points (terminations and bifurcations),
    compute the set of 8D descriptors for each minutia, and return them.
    """
    img_path = example["path"]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"fingerprint_features": []}  # Return empty if image is not found
    # Extract minutiae using your feature extractor (adjust parameters as needed)
    terms, bifurs = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False,
        showResult=False, saveResult=False
    )
    # Get the list of descriptors without averaging them
    descriptors = build_retina_descriptor(terms, bifurs)
    return {"fingerprint_features": descriptors}

# ------------------------------
# 4) BUILD HF DATASET AND DEFINE A COLLATOR
# ------------------------------
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # For training with a classifier, one might need a fixed-length vector.
        # However, for matching we retain the list of descriptors.
        # Here we stack the fingerprint_features (if available) for further processing.
        x = [torch.tensor(f["fingerprint_features"], dtype=torch.float32) 
             if len(f["fingerprint_features"]) > 0 else torch.zeros((1, 8), dtype=torch.float32)
             for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return {"fingerprint_features": x, "labels": labels}

print("Loading datasets...")
# Check if the preprocessed HF datasets exist on disk; if not, build them.
if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) and os.path.exists(test_dataset_path):
    print("Loading existing datasets from disk...")
    train_dataset = load_from_disk(train_dataset_path)
    val_dataset = load_from_disk(val_dataset_path)
    test_dataset = load_from_disk(test_dataset_path)
else:
    print("Building new datasets from", data_dir)
    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    random.seed(42)
    # For each class, split images into training, validation, and test sets
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
        images = [os.path.join(class_dir, f) for f in images]
        random.shuffle(images)
        # Here we use 17 images for training, 4 for validation, 4 for testing per class
        train_imgs = images[:17]
        val_imgs = images[17:21]
        test_imgs = images[21:25]
        label = label2id[class_name]
        train_files.extend(train_imgs)
        val_files.extend(val_imgs)
        test_files.extend(test_imgs)
        train_labels.extend([label] * len(train_imgs))
        val_labels.extend([label] * len(val_imgs))
        test_labels.extend([label] * len(test_imgs))
    os.makedirs(dataset_dir, exist_ok=True)
    # Create dictionaries for the dataset
    train_dict = {"path": train_files, "label": train_labels}
    val_dict = {"path": val_files, "label": val_labels}
    test_dict = {"path": test_files, "label": test_labels}
    # Define the dataset features (schema)
    features = Features({
        "path": Value("string"),
        "label": ClassLabel(num_classes=len(class_names), names=class_names),
    })
    train_dataset = Dataset.from_dict(train_dict).cast(features)
    val_dataset = Dataset.from_dict(val_dict).cast(features)
    test_dataset = Dataset.from_dict(test_dict).cast(features)
    # Map each example to extract its list of 8D descriptors
    train_dataset = train_dataset.map(extract_fingerprint_features)
    val_dataset = val_dataset.map(extract_fingerprint_features)
    test_dataset = test_dataset.map(extract_fingerprint_features)
    # Remove the image path column (no longer needed)
    train_dataset = train_dataset.remove_columns(["path"])
    val_dataset = val_dataset.remove_columns(["path"])
    test_dataset = test_dataset.remove_columns(["path"])
    # Save the datasets to disk for future use
    train_dataset.save_to_disk(train_dataset_path)
    val_dataset.save_to_disk(val_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)

# ------------------------------
# 5) ENROLLMENT AND MATCHING FUNCTIONS
# ------------------------------

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two 1-D numpy arrays.
    Returns 0 if either vector is all zeros.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def match_descriptor_sets(query_descs, enrolled_descs):
    """
    Given a set of descriptors from a query image and a set from an enrolled image,
    compute a matching score.
    
    For each descriptor in the query, find the maximum cosine similarity to any enrolled descriptor.
    Then average these maximum similarities to yield an overall matching score.
    
    Returns a score between -1 and 1.
    """
    if len(query_descs) == 0 or len(enrolled_descs) == 0:
        return 0.0
    scores = []
    for q in query_descs:
        sims = [cosine_similarity(np.array(q), np.array(e)) for e in enrolled_descs]
        scores.append(max(sims) if sims else 0.0)
    return np.mean(scores)

def enroll_dataset(dataset):
    """
    Enroll each sample in the dataset by storing its label and list of descriptors.
    Returns a list of enrollment templates.
    Each template is a dictionary with keys "label" and "descriptors".
    """
    enrollment = []
    for example in dataset:
        enrollment.append({
            "label": example["label"],
            "descriptors": example["fingerprint_features"]
        })
    return enrollment

def identify_query(query_example, enrolled_templates):
    """
    Given a query example (with its list of descriptors), compare it to each enrolled template.
    The matching score is computed using cosine similarity between descriptor sets.
    Returns the predicted label and the best matching score.
    """
    query_descs = query_example["fingerprint_features"]
    best_score = -1.0
    best_label = None
    for temp in enrolled_templates:
        score = match_descriptor_sets(query_descs, temp["descriptors"])
        if score > best_score:
            best_score = score
            best_label = temp["label"]
    return best_label, best_score

# Enroll the training dataset: build a database of enrolled templates.
print("Enrolling training dataset...")
enrolled_templates = enroll_dataset(train_dataset)

# ------------------------------
# 6) EVALUATION VIA MATCHING
# ------------------------------
print("Evaluating on test dataset...")
correct = 0
total = 0
results = []
# Iterate over each test sample, perform identification via matching
for i, example in enumerate(test_dataset):
    print("Processing test sample:", i)
    
    true_label = example["label"]
    predicted_label, score = identify_query(example, enrolled_templates)
    results.append((true_label, predicted_label, score))
    total += 1
    if predicted_label == true_label:
        print("Correctly identified:", id2label[true_label], "as", id2label[predicted_label])
        print("Score:", score)
        correct += 1
    else:
        print("Misidentified:", id2label[true_label], "as", id2label[predicted_label])
        print("Score:", score)
        print("True label:", id2label[true_label])
        print("Predicted label:", id2label[predicted_label])