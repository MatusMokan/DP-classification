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
import fingerprint_feature_extractor  # your existing library for minutiae
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------
# 1) CONFIG & DEVICE
# ------------------------------------------------
with open('config.json', 'r') as f:
    config = json.load(f)
models = config["models"]
selected_model = config["selected_model"]
model_name = models[selected_model]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------
# 2) Setup data_dir, read class_names
# ------------------------------------------------
data_dir = "dataset/onDrive-divided-cropped-augmented"
class_names = sorted(
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
)
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}
print(f"Found {len(class_names)} classes: {class_names}")

# Paths for saving HF Datasets
dataset_dir = "dataset/fingerprint_model"
train_dataset_path = os.path.join(dataset_dir, "train_dataset")
val_dataset_path   = os.path.join(dataset_dir, "val_dataset")
test_dataset_path  = os.path.join(dataset_dir, "test_dataset")


# ------------------------------------------------
# Minutiae Feature Helpers
# ------------------------------------------------
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_degrees(x1, y1, x2, y2):
    """
    Returns angle in degrees from (x1, y1) to (x2, y2), in range [-180, 180].
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def compute_4NN_descriptor(minutia, all_minutiae, k_neighbors=4):
    """
    For one minutia, find the 4 nearest neighbors among 'all_minutiae'.
    Build a 1×8 descriptor: [d1, d2, d3, d4, angle1, angle2, angle3, angle4].
    """
    cx, cy = minutia.locX, minutia.locY

    # Distances to every other minutia
    dist_list = []
    for m in all_minutiae:
        if m is minutia:
            continue
        dist = euclidean_distance(cx, cy, m.locX, m.locY)
        dist_list.append((dist, m))

    # Sort by distance, pick up to 4 neighbors
    dist_list.sort(key=lambda x: x[0])
    four_nearest = dist_list[:k_neighbors]

    distances = []
    angles = []
    for dist, neigh in four_nearest:
        nx, ny = neigh.locX, neigh.locY
        ang = angle_degrees(cx, cy, nx, ny)
        distances.append(dist)
        angles.append(ang)

    # If less than 4 neighbors, pad with zeros
    while len(distances) < 4:
        distances.append(0.0)
        angles.append(0.0)

    return distances + angles  # 8-element list

def build_retina_descriptor(terminations, bifurcations):
    """
    Combine endings & bifurcations into one list.
    For each minutia, compute 4NN descriptor.
    Then average across all minutiae -> final 8D vector.
    If no minutiae, return zeros.
    """
    all_min = terminations + bifurcations
    if len(all_min) == 0:
        return [0.0]*8

    descriptors = []
    for m in all_min:
        desc = compute_4NN_descriptor(m, all_min, k_neighbors=4)
        descriptors.append(desc)

    # shape [num_points, 8]
    arr = np.array(descriptors)
    mean_desc = arr.mean(axis=0)
    return mean_desc.tolist()

def extract_fingerprint_features(example):
    """
    Extracts the 4NN-based 1×8 descriptor for each minutia,
    then averages them into a single 8D descriptor per image.
    """
    img_path = example["path"]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"fingerprint_features": [0.0]*8}

    terms, bifurs = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False,
        showResult=False, saveResult=False
    )
    retina_vec = build_retina_descriptor(terms, bifurs)
    return {"fingerprint_features": retina_vec}


# ------------------------------------------------
# 3) Hugging Face Dataset + Collator
# ------------------------------------------------
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # “pixel_values” are the feature vectors
        x = [torch.tensor(f["fingerprint_features"], dtype=torch.float32) for f in features]
        pixel_values = torch.stack(x, dim=0)
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


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

    # Example approach: 3 train, 1 val, 1 test => 5 images max per class
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
        train_labels.extend([label]*len(train_imgs))
        val_labels.extend([label]*len(val_imgs))
        test_labels.extend([label]*len(test_imgs))

    os.makedirs(dataset_dir, exist_ok=True)

    train_dict = {"path": train_files, "label": train_labels}
    val_dict   = {"path": val_files,   "label": val_labels}
    test_dict  = {"path": test_files,  "label": test_labels}

    features = Features({
        "path":  Value("string"),
        "label": ClassLabel(num_classes=len(class_names), names=class_names),
    })

    train_dataset = Dataset.from_dict(train_dict).cast(features)
    val_dataset   = Dataset.from_dict(val_dict).cast(features)
    test_dataset  = Dataset.from_dict(test_dict).cast(features)

    # Map => build the 8D descriptor for each image
    train_dataset = train_dataset.map(extract_fingerprint_features)
    val_dataset   = val_dataset.map(extract_fingerprint_features)
    test_dataset  = test_dataset.map(extract_fingerprint_features)

    # Remove path column
    train_dataset = train_dataset.remove_columns(["path"])
    val_dataset   = val_dataset.remove_columns(["path"])
    test_dataset  = test_dataset.remove_columns(["path"])

    # Save to disk
    train_dataset.save_to_disk(train_dataset_path)
    val_dataset.save_to_disk(val_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)


# ------------------------------------------------
# 4) Simple PyTorch Model for Classification
# ------------------------------------------------
class SimpleFingerprintModel(nn.Module):
    def __init__(self, num_features=8, num_classes=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )

    def forward(self, pixel_values, labels=None):
        logits = self.net(pixel_values)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

class ComplexFingerprintModel(nn.Module):
    def __init__(self, num_features=8, num_classes=20):
        super().__init__()
        # Adding more layers and neurons for increased capacity
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, pixel_values, labels=None):
        logits = self.net(pixel_values)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

class HFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values, labels=labels)
        loss, logits = outputs["loss"], outputs["logits"]
        return {"loss": loss, "logits": logits}


# 5) Instantiate the model
num_features = 8  # we produce an 8D feature from the 4NN approach
model = ComplexFingerprintModel(num_features=num_features, num_classes=len(class_names)).to(device)
hf_model = HFWrapper(model).to(device)

# 6) Prepare Collator
@dataclass
class FingerprintCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        new_x = []
        labels = []
        for f in features:
            arr = f["fingerprint_features"]
            # Expect arr of length=8
            if len(arr) != 8:
                arr = arr[:8] + [0.0]*(8-len(arr)) if len(arr)<8 else arr[:8]
            new_x.append(torch.tensor(arr, dtype=torch.float32))
            labels.append(f["label"])
        pixel_values = torch.stack(new_x, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

collator = FingerprintCollator()

# 7) TrainingArguments & Trainer
training_args = TrainingArguments(
    output_dir='./results_fingerprint',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    remove_unused_columns=False,
    save_total_limit=2,
)

metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# 8) Train & Evaluate
trainer.train()
metrics = trainer.evaluate(test_dataset)
print("Test set metrics:", metrics)