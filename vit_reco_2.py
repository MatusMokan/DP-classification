import os
import random
from datasets import Dataset, Features, ClassLabel, Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.transforms import (
    Compose, RandomResizedCrop, CenterCrop, Resize, ToTensor, Normalize, 
    RandomHorizontalFlip, RandomRotation, ColorJitter, RandomVerticalFlip, RandomAffine
)
import numpy as np
from evaluate import load as load_metric
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image as PILImage
import torchvision.transforms as T

# conda deactivate
# conda activate /Users/moky/School/4rocnik/DP/ImageReco/env

# Define the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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
# Set seed for reproducibility
# -------------------------
random.seed(42)
np.random.seed(42)

# -------------------------
# Prepare dataset directory and labels
# -------------------------
# data_dir = 'dataset/images-divided'
data_dir = 'dataset/onDrive-divided-cropped'

print("Getting class names...")
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {len(class_names)} classes:")
print(class_names)

# Mapping from class names to labels
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}

# Prepare file paths and labels (3 for train, 1 for val, 1 for test per class)
train_files, val_files, test_files = [], [], []
train_labels, val_labels, test_labels = [], [], []

print("Preparing file paths and labels...")
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    print(class_dir)
    images = [img for img in os.listdir(class_dir) if (img.endswith('.JPG') or img.endswith('.png'))]
    images = [os.path.join(class_dir, img) for img in images]
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
print("File paths and labels prepared.")

# Create datasets
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
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = image_processor.size['height']

# -------------------------
# Custom CLAHE Transform (remains the same)
# -------------------------
class ToGrayCLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    def __call__(self, image):
        # Convert to grayscale and apply CLAHE
        gray = np.array(image.convert('L'))
        clahe_img = self.clahe.apply(gray)
        img_3ch = np.stack([clahe_img] * 3, axis=-1)
        return PILImage.fromarray(img_3ch)

# -------------------------
# Define separate transform pipelines for training and evaluation
# -------------------------
# Training transform: uses RandomResizedCrop, random flips, rotations, color jitter, and CLAHE
train_transform = Compose([
    # CenterCrop((size, size)),
    Resize((size, size)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2),
    # ToGrayCLAHE(clipLimit=2.0, tileGridSize=(8,8)),
    ToTensor(),
    normalize,
])



# train_transform = T.Compose([
#     T.RandomHorizontalFlip(p=0.5),
#     T.RandomVerticalFlip(p=0.5),
#     T.RandomRotation(degrees=15),                # small random rotation, e.g. ±15°
#     T.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0)),  # random crop zoom-in
#     T.ColorJitter(brightness=0.2, contrast=0.2), # adjust brightness/contrast
#     T.ToTensor(),        # convert to tensor for model input
#     # (Optional) Add noise via a custom transform:
#     # T.Lambda(lambda img: img + 0.01 * torch.randn_like(img))
# ])

# Evaluation transform: uses deterministic CenterCrop and Resize, then normalizes
eval_transform = Compose([
    # CenterCrop((size, size)),
    Resize((size, size)),
    ToTensor(),
    normalize,
])

def train_transforms_fn(examples):
    # Directly process the image since it's already a PIL Image.
    pixel_values = [train_transform(img.convert("RGB")) for img in examples["image"]]
    return {"pixel_values": pixel_values, "label": examples["label"]}

def eval_transforms_fn(examples):
    # Process the images and keep the original file path.
    pixel_values = [eval_transform(img.convert("RGB")) for img in examples["image"]]
    return {
        "pixel_values": pixel_values,
        "label": examples["label"],
        "image_path": examples["image"]  # Preserve original file path
    }

print("Applying transforms to datasets...")
train_dataset = train_dataset.with_transform(train_transforms_fn)
val_dataset = val_dataset.with_transform(eval_transforms_fn)
test_dataset = test_dataset.with_transform(eval_transforms_fn)
print("Transforms applied.")

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

# -------------------------
# Load Model and Training Setup
# -------------------------
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.05,  # Add this
    attention_probs_dropout_prob=0.05  # Add this
)

print("Model loaded.")
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=25,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    remove_unused_columns=False,
    save_total_limit=3,

    # learning_rate=5e4,
    # gradient_accumulation_steps=4,
    # weight_decay=0.01,
    # warmup_ratio=0.1,

)

# training_args = TrainingArguments(
#     output_dir='./results',
#     remove_unused_columns=False,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=4,
#     per_device_eval_batch_size=16,
#     num_train_epochs=20,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     save_total_limit=3,
#     logging_dir='./logs',
# )

print("Loading metrics...")
metric = load_metric('accuracy')
misclassified_images = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=preds, references=labels)
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred != label:
            misclassified_images.append((i, label, pred))
    return accuracy

print("Metrics loaded.")

print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=CustomDataCollator(),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
print("Trainer created.")

print("Starting training...")
trainer.train()
print("Training completed.")


# (Optional) Visualization and prediction code remains the same.



def plot_learning_curves(log_history):
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

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(val_epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot validation accuracy (if available)
    if val_acc:
        plt.subplot(1, 2, 2)
        plt.plot(val_epochs, val_acc, label='Validation Accuracy', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.legend()

    plt.tight_layout()
    plt.show()

# After training is complete:
plot_learning_curves(trainer.state.log_history)

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
    
    # Use the original image saved under "image_path". If it's already a PIL Image, use it directly.
    orig = test_example['image_path']
    if isinstance(orig, str):
        true_image = PILImage.open(orig).convert('RGB')
    else:
        true_image = orig.convert('RGB')
    true_image = np.array(true_image) / 255.0  # Scale to [0, 1] for display

    # Get the transformed image (which is what the model sees)
    test_image_tensor = test_example['pixel_values'].to(device)
    predicted_image_array = unnormalize_image(test_image_tensor, mean_arr, std_arr)
    
    with torch.no_grad():
        outputs = model(test_image_tensor.unsqueeze(0))
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = id2label[predicted_class_idx]
        true_label = id2label[test_example['label']]
    
    # Append a tuple containing (true_image, predicted_image, true_label, predicted_label)
    images_and_labels.append((true_image, predicted_image_array, true_label, predicted_label))
# -------------------------
# Interactive Prediction Visualization
# -------------------------
def update_plot(index):
    true_image, predicted_image, true_label, predicted_label = images_and_labels[index]
    
    # Display the original true image in the left subplot
    axes[0].imshow(true_image)
    axes[0].set_title(f"True Label: {true_label}")
    axes[0].axis('off')
    # Display the processed (predicted) image in the right subplot
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