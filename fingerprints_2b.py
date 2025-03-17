import os
import random
import json
import torch
import numpy as np
from datasets import Dataset, Features, ClassLabel, Value, load_from_disk
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from typing import Any, Dict, List
from evaluate import load as load_metric
import fingerprint_feature_extractor  # your existing library for minutiae extraction
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F

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
dataset_dir = "dataset/fingerprint_model_b"
train_dataset_path = os.path.join(dataset_dir, "train_dataset")
val_dataset_path   = os.path.join(dataset_dir, "val_dataset")
test_dataset_path  = os.path.join(dataset_dir, "test_dataset")

# ------------------------------------------------
# 3) MINUTIAE FEATURE HELPER FUNCTIONS
# ------------------------------------------------
def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_degrees(x1, y1, x2, y2):
    """
    Returns the angle (in degrees) from (x1,y1) to (x2,y2) in the range [-180, 180].
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def compute_4NN_descriptor(minutia, all_minutiae, k_neighbors=4):
    """
    For a given minutia, find its 4 nearest neighbors (using Euclidean distance)
    among all_minutiae and compute an 8-D descriptor:
      [d1, d2, d3, d4, angle1, angle2, angle3, angle4].
    """
    cx, cy = minutia.locX, minutia.locY

    # Compute distances from this minutia to every other minutia.
    dist_list = []
    for m in all_minutiae:
        if m is minutia:
            continue
        dist = euclidean_distance(cx, cy, m.locX, m.locY)
        dist_list.append((dist, m))

    # Sort by distance and take up to 4 nearest neighbors.
    dist_list.sort(key=lambda x: x[0])
    four_nearest = dist_list[:k_neighbors]

    distances = []
    angles = []
    for dist, neigh in four_nearest:
        nx, ny = neigh.locX, neigh.locY
        ang = angle_degrees(cx, cy, nx, ny)
        distances.append(dist)
        angles.append(ang)

    # If fewer than 4 neighbors, pad with zeros.
    while len(distances) < 4:
        distances.append(0.0)
        angles.append(0.0)

    return distances + angles  # Returns an 8-element list.

def build_retina_descriptor_set(terminations, bifurcations):
    """
    Instead of averaging, build and return the full set of 8-D descriptors (one per minutia)
    for the given terminations and bifurcations.
    If no minutiae are found, return an empty list.
    """
    all_min = terminations + bifurcations
    if len(all_min) == 0:
        return []
    descriptors = []
    for m in all_min:
        desc = compute_4NN_descriptor(m, all_min, k_neighbors=4)
        descriptors.append(desc)
    return descriptors  # A list of 8-D vectors.

def extract_fingerprint_features(example):
    """
    For a given image (provided by its file path in example["path"]),
    load the image, extract minutiae using the provided library, and compute
    the full set of 8-D descriptors. (This function retains the complete set.)
    """
    img_path = example["path"]
    print(f"Processing image: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        return {"fingerprint_features": []}
    # Extract minutiae: terminations and bifurcations.
    terms, bifurs = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False,
        showResult=False, saveResult=False
    )
    print(f"Found {len(terms)} terminations and {len(bifurs)} bifurcations")
    # Compute the full set of 8-D descriptors.
    descriptors = build_retina_descriptor_set(terms, bifurs)

    # added
    # Add normalization
    if descriptors:
        descriptors = np.array(descriptors)
        # Normalize distances
        mean_dist = np.mean(descriptors[:, :4])
        std_dist = np.std(descriptors[:, :4]) + 1e-6
        descriptors[:, :4] = (descriptors[:, :4] - mean_dist) / std_dist
        
        # Normalize angles (circular features)
        descriptors[:, 4:] = np.sin(np.radians(descriptors[:, 4:]))
        
        descriptors = descriptors.tolist()
    
    print(f"Extracted {len(descriptors)} descriptors")
    return {"fingerprint_features": descriptors}

    print(f"Extracted {len(descriptors)} descriptors")
    return {"fingerprint_features": descriptors}

# ------------------------------------------------
# 4) BUILD OR LOAD HF DATASET & COLLATOR FOR SET-BASED DATA
# ------------------------------------------------
# We want to store the full set (variable-length) of descriptors per image.
# We use a custom collator that pads each set to the maximum number in the batch.
@dataclass
class SetCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_descriptors = []
        lengths = []
        labels = []
        # For each sample, convert its list of descriptors (each descriptor is an 8-D list)
        # into a tensor of shape (num_descriptors, 8). If empty, use a single zero vector.
        for f in features:
            desc_list = f["fingerprint_features"]
            if not desc_list:
                desc_tensor = torch.zeros((1, 8), dtype=torch.float32)
                lengths.append(1)
            else:
                desc_tensor = torch.tensor(desc_list, dtype=torch.float32)
                lengths.append(desc_tensor.shape[0])
            batch_descriptors.append(desc_tensor)
            labels.append(f["label"])
        # Pad the list of descriptor tensors so that all samples have the same number of descriptors.
        padded = torch.nn.utils.rnn.pad_sequence(batch_descriptors, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": padded, "lengths": lengths, "labels": labels}

# Use the custom collator for set-based inputs.
collator = SetCollator()

# Load preprocessed datasets if they exist; otherwise, build them.
if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) and os.path.exists(test_dataset_path):
    print("Loading existing datasets from disk...")
    train_dataset = load_from_disk(train_dataset_path)
    val_dataset   = load_from_disk(val_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)
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
# 5) DEFINE A DEEPSET-STYLE FINGERPRINT MODEL
# ------------------------------------------------
# Since we are now working with sets of 8-D descriptors, we use a model that processes each set.
# Here we define a simple DeepSet-like model that applies a learned function phi() to each descriptor,
# aggregates via both mean and max pooling, and then uses a classifier (rho) to predict the class.
class DeepSetFingerprintModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, out_dim=128, num_classes=20):
        super().__init__()
        # Increase model capacity
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Add BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increase dropout
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, pixel_values, lengths, labels=None):
        # Handle BatchNorm1d with varying sequence lengths
        B, N, D = pixel_values.shape
        phi_x = pixel_values.view(-1, D)
        phi_x = self.phi(phi_x).view(B, N, -1)
        
        # Create mask for valid descriptors
        mask = torch.arange(N, device=pixel_values.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        # Apply attention for weighted pooling
        att_weights = self.attention(phi_x)
        att_weights = att_weights * mask
        att_weights = att_weights / (att_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # Weighted average pooling
        pooled = (phi_x * att_weights).sum(dim=1)
        
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

# Wrapper to ensure the Trainer expects a dictionary output.
class HFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, lengths, labels=None):
        outputs = self.model(pixel_values, lengths, labels=labels)
        loss, logits = outputs["loss"], outputs["logits"]
        return {"loss": loss, "logits": logits}

# 6) Instantiate the model.
num_features = 8  # Each descriptor is 8-D.
num_classes = len(class_names)
model = DeepSetFingerprintModel(input_dim=num_features, hidden_dim=32, out_dim=64, num_classes=num_classes).to(device)
hf_model = HFWrapper(model).to(device)

# ------------------------------------------------
# 7) TRAINING ARGUMENTS & SETUP
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir='./results_fingerprint',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=100,  # Train longer
    per_device_train_batch_size=8,  # Increase if memory allows
    per_device_eval_batch_size=8,
    learning_rate=2e-5,  # Lower learning rate
    weight_decay=0.01,  # Add weight decay
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    remove_unused_columns=False,
    save_total_limit=2,
    warmup_steps=200,  # Add warmup
)

# We'll use an accuracy metric from the evaluate library.
metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# ------------------------------------------------
# 8) SET UP THE TRAINER WITH OUR SET COLLATOR
# ------------------------------------------------
trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# ------------------------------------------------
# 9) TRAIN & EVALUATE THE MODEL
# ------------------------------------------------
trainer.train()
metrics = trainer.evaluate(test_dataset)
print("Test set metrics:", metrics)

