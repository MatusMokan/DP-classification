import os
import random
import json
import torch
import numpy as np
from datasets import Dataset, Features, ClassLabel, Value, load_from_disk
from torch import nn
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List
from evaluate import load as load_metric
from torchvision import models
from icecream import ic
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
import torchvision.models as models


# ------------------------------------------------
# 1) Load config (if you have config.json)
# ------------------------------------------------
with open('config.json', 'r') as f:
    config = json.load(f)

# Example model name from config
# models = config["models"]
# selected_model = config["selected_model"]
# model_name = models[selected_model]

# ------------------------------------------------
# 2) Define the device
# ------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------
# 3) Collator (stacks feature vectors and labels)
# ------------------------------------------------
# @dataclass
# class CustomDataCollator:
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         # “pixel_values” here are actually your minutiae vectors
#         x = [torch.tensor(f["fingerprint_features"], dtype=torch.float32) for f in features]
#         # Stack into [batch_size, feature_dim]
#         pixel_values = torch.stack(x, dim=0)
#         labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
#         return {
#             "pixel_values": pixel_values,
#             "labels": labels
#         }

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

        locX_hist = np.histogram(locX, bins=10, range=(0, 512))[0]
        locY_hist = np.histogram(locY, bins=10, range=(0, 512))[0]
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
    data_dir = "dataset/onDrive-divided"  # adjust as needed
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
    train_dataset = train_dataset.map(extract_fingerprint_features)
    val_dataset   = val_dataset.map(extract_fingerprint_features)
    test_dataset  = test_dataset.map(extract_fingerprint_features)

    # We don’t really use the image field anymore, so remove it
    train_dataset = train_dataset.remove_columns(["path"])
    val_dataset   = val_dataset.remove_columns(["path"])
    test_dataset  = test_dataset.remove_columns(["path"])

    # Save datasets to disk
    train_dataset.save_to_disk(train_dataset_path)
    val_dataset.save_to_disk(val_dataset_path)
    test_dataset.save_to_disk(test_dataset_path)


ic(train_dataset[0]) 

# Determine vector size from dataset
vector_size = len(train_dataset[0]["fingerprint_features"])

print(f"Vector size: {vector_size}")

# ------------------------------------------------
# 3) Define Model
# ------------------------------------------------
class ResNetForVectors(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ResNetForVectors, self).__init__()
        # Load pretrained ResNet
        self.resnet = models.resnet50(pretrained=True)
        
        # Replace convolutional layers with input layer for vectors
        self.resnet.fc = nn.Sequential(
            nn.Linear(input_size, 512),  # Match input_size to vector size
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Output layer
        )

    def forward(self, features, labels=None):
        logits = self.resnet.fc(features)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

ic(len(train_dataset.features["label"].names))
model = ResNetForVectors(input_size=vector_size, num_classes=len(train_dataset.features["label"].names))
model.to(device)

# ------------------------------------------------
# 4) Define Custom Data Collator
# ------------------------------------------------

# @dataclass
# class VectorCollator:
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         vectors = torch.tensor([f["fingerprint_features"] for f in features], dtype=torch.float32)
#         labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
#         return {"fingerprint_features": vectors, "labels": labels}



@dataclass
class VectorCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Debug: Check the first batch sample
        print(f"Batch sample: {features[:1]}")
        ic(features)
        # Ensure the correct key is accessed
        try:
            vectors = torch.tensor([f["fingerprint_features"] for f in features], dtype=torch.float32)
        except KeyError as e:
            print(f"KeyError: {e}")
            print(f"Features: {features}")
            raise e

        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)

        return {"pixel_values": vectors, "labels": labels}
    
collator = VectorCollator()

# ------------------------------------------------
# 5) Training Arguments and Trainer
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir='./results_resnet_vectors',
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

ic(train_dataset)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ------------------------------------------------
# 6) Train and Evaluate
# ------------------------------------------------
trainer.train()
metrics = trainer.evaluate(test_dataset)
ic(metrics)