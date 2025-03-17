
import os
import random
import json
import torch
import numpy as np
from datasets import Dataset, Features, ClassLabel, Sequence, Value, load_from_disk
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from typing import Any, Dict, List
from evaluate import load as load_metric
import fingerprint_feature_extractor
import cv2
from icecream import ic


# ------------------------------------------------
# 1) Load config (if you have config.json)
# ------------------------------------------------
with open('config.json', 'r') as f:
    config = json.load(f)

# Example model name from config
models = config["models"]
selected_model = config["selected_model"]
model_name = models[selected_model]

# ------------------------------------------------
# 2) Define the device
# ------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------
# 3) Collator (stacks feature vectors and labels)
# ------------------------------------------------
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # “pixel_values” here are actually your minutiae vectors
        x = [torch.tensor(f["fingerprint_features"], dtype=torch.float32) for f in features]
        # Stack into [batch_size, feature_dim]
        pixel_values = torch.stack(x, dim=0)
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# ------------------------------------------------
# 5) Define your feature-extraction function
# ------------------------------------------------

def aggregate_minutiae_features(terminations, bifurcations):
    """
    Aggregates minutiae features into a fixed-length vector using histograms
    and statistical summaries for Termination and Bifurcation features.
    """

    def compute_features(features, name_prefix):
        locX = [f.locX for f in features]
        locY = [f.locY for f in features]
        orientations = [o for f in features for o in f.Orientation]

        locX_hist = np.histogram(locX, bins=10, range=(0, 1095))[0]
        locY_hist = np.histogram(locY, bins=10, range=(0, 1095))[0]
        orientation_hist = np.histogram(orientations, bins=10, range=(-180, 180))[0]

        locX_hist = locX_hist / np.sum(locX_hist) if np.sum(locX_hist) > 0 else locX_hist
        locY_hist = locY_hist / np.sum(locY_hist) if np.sum(locY_hist) > 0 else locY_hist
        orientation_hist = orientation_hist / np.sum(orientation_hist) if np.sum(orientation_hist) > 0 else orientation_hist

        locX_stats = [np.mean(locX), np.std(locX), np.min(locX), np.max(locX)] if locX else [0, 0, 0, 0]
        locY_stats = [np.mean(locY), np.std(locY), np.min(locY), np.max(locY)] if locY else [0, 0, 0, 0]
        orientation_stats = [np.mean(orientations), np.std(orientations), np.min(orientations), np.max(orientations)] if orientations else [0, 0, 0, 0]

        combined = np.concatenate([locX_hist, locY_hist, orientation_hist, locX_stats, locY_stats, orientation_stats])
        return combined

    term_features = compute_features(terminations, "termination")
    bifur_features = compute_features(bifurcations, "bifurcation")
    aggregated_features = np.concatenate([term_features, bifur_features])

    return aggregated_features

def extract_fingerprint_features2(example):
    """
    Extracts aggregated minutiae features from a fingerprint image.
    """
    img_path = example["path"]
    img = cv2.imread(img_path, 0)
    if img is None:
        return {"fingerprint_features": [0.0] * 60}  # Replace 60 with the chosen vector size

    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False
    )

    aggregated_features = aggregate_minutiae_features(FeaturesTerminations, FeaturesBifurcations)
    return {"fingerprint_features": aggregated_features.tolist()}

def extract_fingerprint_features(example):
    # Read image
    img_path = example["path"]
    img = cv2.imread(img_path, 0)
    if img is None:
        # If no image found, return zeros
        return {"fingerprint_features": [0.0]}

    # Extract minutiae
    # You’ll get something like (Terminations, Bifurcations)
    # Each is a list of MinutiaeFeature objects
    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False
    )

    # Convert them into a numeric vector
    # For example, gather (locX, locY, Orientation, Type)
    # This is just one simplistic approach
    feats = []
    for f in FeaturesTerminations:
        feats += [f.locX, f.locY]
        # Orientation is a list, you could add e.g. mean orientation
        if isinstance(f.Orientation, list) and f.Orientation:
            feats.append(float(f.Orientation[0])) 
        else:
            feats.append(0.0)

    for f in FeaturesBifurcations:
        feats += [f.locX, f.locY]
        if isinstance(f.Orientation, list) and f.Orientation:
            feats.append(float(f.Orientation[0]))
        else:
            feats.append(0.0)

    # Return your final vector
    return {"fingerprint_features": feats}


# Define paths to save datasets
dataset_dir = "dataset/fingerprint"
train_dataset_path = os.path.join(dataset_dir, "train_dataset")
val_dataset_path = os.path.join(dataset_dir, "val_dataset")
test_dataset_path = os.path.join(dataset_dir, "test_dataset")

# Check if datasets already exist
if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) and os.path.exists(test_dataset_path):
    print("Loading existing datasets...")
    train_dataset = load_from_disk(train_dataset_path)
    val_dataset = load_from_disk(val_dataset_path)
    test_dataset = load_from_disk(test_dataset_path)
else:
    print("Building new datasets...")
    # ------------------------------------------------
    # 4) Create or load your dataset
    # ------------------------------------------------
    data_dir = "dataset/onDrive-divided-cropped"
    print("Getting class names...")
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    print(f"Found {len(class_names)} classes: {class_names}")

    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    random.seed(42)
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        images = [os.path.join(class_dir, f) for f in images]
        random.shuffle(images)
        train_imgs = images[:3]
        val_imgs = images[3:4]
        test_imgs = images[4:5]
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


    # ------------------------------------------------
    # 6) Build the HF Datasets
    # ------------------------------------------------
    features = Features({
        "path": Value("string"),
        "label": ClassLabel(num_classes=len(class_names), names=class_names),
    })

    train_dataset = Dataset.from_dict(train_dict).cast(features)
    val_dataset   = Dataset.from_dict(val_dict).cast(features)
    test_dataset  = Dataset.from_dict(test_dict).cast(features)

    # Map to extract numeric fingerprint vectors
    train_dataset = train_dataset.map(extract_fingerprint_features2)
    val_dataset   = val_dataset.map(extract_fingerprint_features2)
    test_dataset  = test_dataset.map(extract_fingerprint_features2)

    # We don’t really use the image field anymore, so remove it
    train_dataset = train_dataset.remove_columns(["path"])
    val_dataset   = val_dataset.remove_columns(["path"])
    test_dataset  = test_dataset.remove_columns(["path"])

    # Save datasets to disk
    train_dataset.save_to_disk(train_dataset_path)
    val_dataset.save_to_disk(val_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)



# ------------------------------------------------
# 7) Load or define a model
# ------------------------------------------------
# For a quick test, you can define a simple classification head in PyTorch
# Or you can adapt a Transformer to handle generic inputs. Example below just uses a random feed-forward model.

from torch import nn
import torch.nn.functional as F

class SimpleFingerprintModel(nn.Module):
    def __init__(self, num_features=300, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, pixel_values, labels=None):
        logits = self.net(pixel_values)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        # Ensure loss is a tensor
        return {"loss": loss, "logits": logits}

class HFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, labels=None):
        # Your base model returns a tuple (loss, logits)
        outputs = self.model(pixel_values, labels=labels)
        loss, logits = outputs["loss"], outputs["logits"]
        # The Trainer expects a dict
        return {
            "loss": loss,
            "logits": logits
        }

# Figure out a max dimension to pick for input 
# (the largest length of fingerprint_features you might get)
# For simplicity, choose a big enough dimension:
max_len = 0
for row in train_dataset:
    max_len = max(max_len, len(row["fingerprint_features"]))
    avg_len = sum(len(row["fingerprint_features"]) for row in train_dataset) / len(train_dataset)

ic(max_len)
ic(avg_len)
max_len = len(train_dataset[0]["fingerprint_features"])

model = SimpleFingerprintModel(num_features=max_len)
model.to(device)

# ------------------------------------------------
# 8) Pad or cut your fingerprint vectors inside collator
# ------------------------------------------------
@dataclass
class FingerprintCollator:
    max_len: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pad or truncate fingerprint_features to max_len
        # ic(features)
        new_x = []
        labels = []
        for f in features:
            arr = f["fingerprint_features"]
            if len(arr) < self.max_len:
                arr = arr + [0.0]*(self.max_len - len(arr))
            else:
                arr = arr[:self.max_len]
            new_x.append(torch.tensor(arr, dtype=torch.float32))
            labels.append(f["label"])
        pixel_values = torch.stack(new_x, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

collator = FingerprintCollator(max_len)

# ------------------------------------------------
# 9) Training arguments & Trainer
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir='./results_fingerprint',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=50,
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

# # HF Trainer expects model outputs to have “logits” key
# class HFWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     def forward(self, pixel_values, labels=None):
#         outputs = self.model(pixel_values, labels=labels)
#         logits = outputs["logits"]
#         return (logits,)

hf_model = HFWrapper(model).to(device)
# hf_model = (model).to(device)

ic(train_dataset)

trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

# ------------------------------------------------
# 10) Train and Evaluate
# ------------------------------------------------
trainer.train()
metrics = trainer.evaluate(test_dataset)
ic(metrics)