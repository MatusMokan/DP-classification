import os
import random
from datasets import Dataset, Features, ClassLabel, Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import torchvision.transforms as T
import torch
import numpy as np
from evaluate import load as load_metric
from dataclasses import dataclass
from typing import Any, Dict, List
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold  # <-- for k-fold cross-validation

def plot_learning_curves(log_history, fold):
    # Lists to hold metrics
    train_epochs, train_loss = [], []
    val_epochs, val_loss, val_acc = [], [], []

    # Iterate through the log history and extract metrics
    for entry in log_history:
        if 'loss' in entry and 'epoch' in entry:
            train_epochs.append(entry['epoch'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry and 'epoch' in entry:
            val_epochs.append(entry['epoch'])
            val_loss.append(entry['eval_loss'])
        if 'eval_accuracy' in entry and 'epoch' in entry:
            val_acc.append(entry['eval_accuracy'])

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(val_epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    if val_acc:
        plt.plot(val_epochs, val_acc, label='Validation Accuracy', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.legend()

    plt.tight_layout()
    
    # Create folder if it doesn't exist
    save_folder = "learning_curves"
    os.makedirs(save_folder, exist_ok=True)
    
    save_path = os.path.join(save_folder, f"learning_curve_fold{fold+1}.png")
    plt.savefig(save_path)
    print(f"Learning curve for fold {fold+1} saved to {save_path}")
    plt.close()

# -------------------------
# Configuration
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'google/vit-large-patch32-384'
k_folds = 5  # number of folds to use in cross-validation

# -------------------------
# Custom Data Collator
# -------------------------
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = torch.tensor([feature["label"] for feature in features])
        return {"pixel_values": pixel_values, "labels": labels}

# -------------------------
# Seed for Reproducibility
# -------------------------
random.seed(42)
np.random.seed(42)

# -------------------------
# Prepare Datasets
# -------------------------
data_dir = 'dataset/onDrive-divided-cropped'

# Get class names (subfolders)
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Found classes:", class_names)

# Build label mappings
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}

all_files = []
all_labels = []
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    files = [os.path.join(class_dir, f) 
             for f in os.listdir(class_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file in files:
        all_files.append(file)
        all_labels.append(label2id[class_name])

print(f"Total images: {len(all_files)}")

# Create ONE dataset with all images (we'll split it via k-fold)
all_dict = {'image': all_files, 'label': all_labels}
features = Features({
    'image': Image(),
    'label': ClassLabel(num_classes=len(class_names), names=class_names),
})
full_dataset = Dataset.from_dict(all_dict).cast(features)

print("Full dataset size:", len(full_dataset))

# -------------------------
# Load Image Processor & Define Transforms
# -------------------------
image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
normalize = T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = image_processor.size['height']

# Example transforms
train_transform = T.Compose([
    T.Resize((size, size)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    normalize
])

eval_transform = T.Compose([
    T.Resize((size, size)),
    T.ToTensor(),
    normalize
])

def train_transforms_fn(examples):
    pixel_values = []
    for img in examples["image"]:
        img_pil = img.convert("RGB")
        pixel_values.append(train_transform(img_pil))
    return {"pixel_values": pixel_values, "label": examples["label"]}

def eval_transforms_fn(examples):
    pixel_values = []
    for img in examples["image"]:
        img_pil = img.convert("RGB")
        pixel_values.append(eval_transform(img_pil))
    return {"pixel_values": pixel_values, "label": examples["label"]}

# -------------------------
# Define Metric Computation
# -------------------------
accuracy_metric = load_metric('accuracy')
misclassified_images = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    return acc

# -------------------------
# k-Fold Cross Validation
# -------------------------
print(f"Starting {k_folds}-fold cross-validation...")
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"\n===== Fold {fold+1} / {k_folds} =====")
    
    # Subset the dataset for this fold
    train_subsplit = full_dataset.select(train_ids)
    val_subsplit = full_dataset.select(val_ids)

    # Apply transforms
    train_subsplit = train_subsplit.with_transform(train_transforms_fn)
    val_subsplit = val_subsplit.with_transform(eval_transforms_fn)

    # Load model fresh each fold (or you can reuse if you like, but typically you re-init)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=0.05,
        attention_probs_dropout_prob=0.05
    ).to(device)

    # Training arguments (you could adjust these, or set fewer epochs for each fold)
    training_args = TrainingArguments(
        output_dir=f'./results-fold{fold+1}',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=25,  # fewer epochs in CV to save time
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir=f'./logs-fold{fold+1}',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        remove_unused_columns=False,
        save_total_limit=1,
        learning_rate=5e-5,  # Set a lower learning rate
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subsplit,
        eval_dataset=val_subsplit,
        data_collator=CustomDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train
    trainer.train()

    # Plot and save the learning curves for this fold
    plot_learning_curves(trainer.state.log_history, fold)


    # Evaluate on this fold's validation set
    metrics = trainer.evaluate(val_subsplit)
    fold_acc = metrics["eval_accuracy"]
    print(f"Fold {fold+1} accuracy = {fold_acc:.4f}")

    fold_accuracies.append(fold_acc)

# After all folds
print("\nCross-validation complete!")
print("Fold accuracies:", fold_accuracies)
print("Mean accuracy:", np.mean(fold_accuracies))
print("Std dev:", np.std(fold_accuracies))