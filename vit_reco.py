import os
import random
from datasets import Dataset, Features, ClassLabel, Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
import numpy as np
from evaluate import load as load_metric
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image as PILImage

# conda deactivate
# conda activate /Users/moky/School/4rocnik/DP/ImageReco/env

# Define the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Custom Collator
# -------------------------
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = torch.tensor([feature["label"] for feature in features])
        return {"pixel_values": pixel_values, "labels": labels}

# -------------------------
# Set seed for reproducibility
# -------------------------
random.seed(42)
np.random.seed(42)

# -------------------------
# Prepare dataset paths and labels
# -------------------------

# data_dir = 'dataset/images-divided'
data_dir = 'dataset/onDrive-divided'

print("Getting class names...")
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {len(class_names)} classes:")
print(class_names)

# Mapping from class names to labels
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}

# Prepare file paths and labels
train_files, val_files, test_files = [], [], []
train_labels, val_labels, test_labels = [], [], []

print("Preparing file paths and labels...")
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    print("Processing:", class_dir)
    images = [img for img in os.listdir(class_dir) if (img.endswith('.JPG') or img.endswith('.png'))]
    images = [os.path.join(class_dir, img) for img in images]
    random.shuffle(images)
    # Split images (adjust splits as needed)
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
print("File paths and labels prepared.")
print("Training samples:", len(train_files))
print("Validation samples:", len(val_files))
print("Test samples:", len(test_files))

# -------------------------
# Create Datasets
# -------------------------
print("Creating datasets...")
train_dict = {'image': train_files, 'label': train_labels}
val_dict = {'image': val_files, 'label': val_labels}
test_dict = {'image': test_files, 'label': test_labels}

print("Creating features...")
features = Features({
    'image': Image(), 
    'label': ClassLabel(num_classes=len(class_names), names=class_names)
})

train_dataset = Dataset.from_dict(train_dict).cast(features)
val_dataset = Dataset.from_dict(val_dict).cast(features)
test_dataset = Dataset.from_dict(test_dict).cast(features)
print("Datasets created.")
print("Train dataset length:", len(train_dataset))
print("Validation dataset length:", len(val_dataset))
print("Test dataset length:", len(test_dataset))

# -------------------------
# Load Image Processor
# -------------------------
print("Loading image processor...")
image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast=True)
print("Image processor loaded.")

# -------------------------
# Custom CLAHE Transform
# -------------------------
class ToGrayCLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, image):
        # Convert to grayscale (L mode) then to numpy array
        gray = np.array(image.convert('L'))
        # Apply CLAHE
        clahe_img = self.clahe.apply(gray)
        # Replicate the single channel into 3 channels
        img_3ch = np.stack([clahe_img] * 3, axis=-1)
        return PILImage.fromarray(img_3ch)

# -------------------------
# Define Transforms
# -------------------------
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = image_processor.size['height']

_transform = Compose([
    Resize((size, size)),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToGrayCLAHE(clipLimit=2.0, tileGridSize=(8,8)),  # Apply custom CLAHE transform
    ToTensor(),
    normalize,
])

# Deterministic transform for validation and test
_eval_transform = Compose([
    Resize((size, size)),
    ToTensor(),
    normalize,
])

def train_transforms_fn(examples):
    pixel_values = [_transform(image.convert('RGB')) for image in examples['image']]
    return {"pixel_values": pixel_values, "label": examples["label"]}

def eval_transforms_fn(examples):
    pixel_values = [_eval_transform(image.convert('RGB')) for image in examples['image']]
    return {"pixel_values": pixel_values, "label": examples["label"]}

print("Applying transforms to datasets...")
train_dataset = train_dataset.with_transform(train_transforms_fn)
val_dataset = val_dataset.with_transform(eval_transforms_fn)
test_dataset = test_dataset.with_transform(eval_transforms_fn)
print("Transforms applied.")

# -------------------------
# Load Model
# -------------------------
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
print("Model loaded.")
model.to(device)

# -------------------------
# Training Arguments and Metrics
# -------------------------
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=25,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    remove_unused_columns=False,
    save_total_limit=3,
)

print("Loading metrics...")
metric = load_metric('accuracy')
misclassified_images = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=preds, references=labels)
    # Track misclassified images
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred != label:
            misclassified_images.append((i, label, pred))
    return accuracy

print("Metrics loaded.")

# -------------------------
# Create Trainer and Train
# -------------------------
print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=CustomDataCollator(),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
)
print("Trainer created.")

print("Starting training...")
trainer.train()
print("Training completed.")

# -------------------------
# Evaluate Model
# -------------------------
print("Evaluating model...")
metrics = trainer.evaluate(test_dataset)
print("Evaluation completed.")
print("Metrics:", metrics)

# -------------------------
# Utility to Unnormalize Image
# -------------------------
def unnormalize_image(image_tensor, mean, std):
    image_array = image_tensor.cpu().permute(1, 2, 0).numpy()
    image_array = std * image_array + mean
    image_array = np.clip(image_array, 0, 1)
    return image_array

# -------------------------
# Display Misclassified Images (if any)
# -------------------------
if misclassified_images:
    print("Displaying misclassified images...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for index, true_label, predicted_label in misclassified_images:
        test_example = test_dataset[index]
        test_image_tensor = test_example['pixel_values'].to(device)
        test_image_array = unnormalize_image(test_image_tensor, image_processor.image_mean, image_processor.image_std)
        
        axes[0].imshow(test_image_array)
        axes[0].set_title(f"True Label: {id2label[true_label]}")
        axes[0].axis('off')
        axes[1].imshow(test_image_array)
        axes[1].set_title(f"Predicted Label: {id2label[predicted_label]}")
        axes[1].axis('off')
        plt.show()
else:
    print("No misclassified images found.")

# -------------------------
# Plot Training Loss History
# -------------------------
training_loss = trainer.state.log_history
loss_values = [entry['loss'] for entry in training_loss if 'loss' in entry]
epochs = [entry['epoch'] for entry in training_loss if 'loss' in entry]

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -------------------------
# Visualize an Example from Training Set
# -------------------------
example = train_dataset[0]
image_tensor = example['pixel_values']
image_array = image_tensor.permute(1, 2, 0).numpy()
mean_np = np.array(image_processor.image_mean)
std_np = np.array(image_processor.image_std)
image_array = std_np * image_array + mean_np
image_array = np.clip(image_array, 0, 1)

plt.imshow(image_array)
plt.axis('off')
plt.show()

# -------------------------
# Make Predictions on Test Set
# -------------------------
model.eval()
print("Making predictions...")

mean_arr = np.array(image_processor.image_mean)
std_arr = np.array(image_processor.image_std)
images_and_labels = []

for i, test_example in enumerate(test_dataset):
    print(f"Test sample {i} - True label:", test_example['label'])
    test_image_tensor = test_example['pixel_values'].to(device)
    test_image_array = unnormalize_image(test_image_tensor, mean_arr, std_arr)

    with torch.no_grad():
        outputs = model(test_image_tensor.unsqueeze(0))
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = id2label[predicted_class_idx]
        true_label = id2label[test_example['label']]

    # Use the same test image for visualization
    images_and_labels.append((test_image_array, test_image_array, true_label, predicted_label))

# -------------------------
# Interactive Prediction Visualization
# -------------------------
def update_plot(index):
    true_image, predicted_image, true_label, predicted_label = images_and_labels[index]
    axes[0].imshow(true_image)
    axes[0].set_title(f"True Label: {true_label}")
    axes[0].axis('off')
    axes[1].imshow(predicted_image)
    axes[1].set_title(f"Predicted Label: {predicted_label}")
    axes[1].axis('off')
    fig.canvas.draw()

def on_key(event):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(images_and_labels)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(images_and_labels)
    update_plot(current_index)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
current_index = 0
update_plot(current_index)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()