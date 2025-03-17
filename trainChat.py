import os
import torch
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Image
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, CLAHE, ElasticTransform, GaussianBlur, RandomRotate90, HorizontalFlip, Normalize as AlbNormalize
)
from albumentations.pytorch import ToTensorV2
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback, get_scheduler
)
from sklearn.metrics import classification_report
from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image as PILImage
import numpy as np

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset directory
data_dir = "dataset/images-divided"

# Get class names
print("Getting class names...")
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {len(class_names)} classes: {class_names}")

# Mapping from class names to labels
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}
# Prepare file paths and labels
all_files = []
all_labels = []

print("Preparing file paths and labels...")
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(".JPG")]
    label = label2id[class_name]
    all_files.extend(images)
    all_labels.extend([label] * len(images))
print("File paths and labels prepared.")

# Stratified splitting
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, all_labels, test_size=0.4, stratify=all_labels, random_state=42
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Create datasets
print("Creating datasets...")
train_dict = {'image': train_files, 'label': train_labels}
val_dict = {'image': val_files, 'label': val_labels}
test_dict = {'image': test_files, 'label': test_labels}

features = Features({'image': Image(), 'label': ClassLabel(num_classes=len(class_names), names=class_names)})

train_dataset = Dataset.from_dict(train_dict).cast(features)
val_dataset = Dataset.from_dict(val_dict).cast(features)
test_dataset = Dataset.from_dict(test_dict).cast(features)
print("Datasets created.")

# Debugging: Check a sample after preprocessing
print("Sample after preprocessing:", train_dataset[0])

# Initialize image processor
print("Loading image processor...")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
print("Image processor loaded.")

# Augmentation with Albumentations
print("Defining augmentations...")
train_transform = Compose([
    CLAHE(),
    ElasticTransform(alpha=1),
    GaussianBlur(blur_limit=(3, 7)),
    RandomRotate90(),
    HorizontalFlip(0.5),
    AlbNormalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ToTensorV2(),
])

val_test_transform = Compose([
    AlbNormalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ToTensorV2(),
])

def transforms(examples, transform):
    # Only proceed if "image" is in the batch
    if "image" not in examples:
        return {"pixel_values": [], "label": examples["label"]}

    pixel_values = []
    for img in examples["image"]:
        transformed = transform(image=np.array(img.convert("RGB")))
        pixel_values.append(transformed["image"])
    return {"pixel_values": pixel_values, "label": examples["label"]}

train_dataset = train_dataset.with_transform(lambda examples: transforms(examples, train_transform))
val_dataset = val_dataset.with_transform(lambda examples: transforms(examples, val_test_transform))
test_dataset = test_dataset.with_transform(lambda examples: transforms(examples, val_test_transform))

# Custom collator
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = torch.tensor([feature["label"] for feature in features])
        return {"pixel_values": pixel_values, "labels": labels}

# Define model
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.to(device)
print("Model loaded.")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Updated to avoid deprecation warning
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=25,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=CustomDataCollator(),
    compute_metrics=lambda eval_pred: {
        "accuracy": (preds := np.argmax(eval_pred[0], axis=-1)) == eval_pred[1],
    },
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train and evaluate
print("Training the model...")
trainer.train()
print("Evaluating the model...")
metrics = trainer.evaluate(test_dataset)
print("Final Metrics:", metrics)