import os
import random
import json
from datasets import Dataset, Features, ClassLabel, Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from evaluate import load as load_metric
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter
import matplotlib.pyplot as plt
import numpy as np

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

models = config["models"]
selected_model = config["selected_model"]
model_name = models[selected_model]

# Define the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Add custom collator class
@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Stack the pixel_values
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        # Convert labels to tensor
        labels = torch.tensor([feature["label"] for feature in features])
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define dataset directory
data_dir = 'dataset/onDrive-divided'

# Get class names
print("Getting class names...")
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {len(class_names)} classes.")

# Mapping from class names to labels
label2id = {name: idx for idx, name in enumerate(class_names)}
id2label = {idx: name for name, idx in label2id.items()}

# Prepare file paths and labels
train_files = []
val_files = []
test_files = []
train_labels = []
val_labels = []
test_labels = []

print("Preparing file paths and labels...")
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    images = [img for img in os.listdir(class_dir) if img.endswith('.png')]
    images = [os.path.join(class_dir, img) for img in images]
    # Shuffle images
    random.shuffle(images)
    # Split images
    train_imgs = images[:3]
    val_imgs = images[3:4]
    test_imgs = images[4:5]
    label = label2id[class_name]
    train_files.extend(train_imgs)
    val_files.extend(val_imgs)
    test_files.extend(test_imgs)
    train_labels.extend([label]*len(train_imgs))
    val_labels.extend([label]*len(val_imgs))
    test_labels.extend([label]*len(test_imgs))
print("File paths and labels prepared.")

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

# Load feature extractor and model based on selected model
print(f"Loading feature extractor for {model_name}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
print("Feature extractor loaded.")
print(f"Loading model {model_name}...")
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=len(class_names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
print("Model loaded.")

# Move model to the correct device
model.to(device)

# Define transforms
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
size = 224  # Default size for ResNet models

_transform = Compose([
    Resize((size, size)),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    normalize,
])

def transforms(examples):
    # Convert images to RGB and apply transforms
    pixel_values = [_transform(image.convert('RGB')) for image in examples['image']]
    # Return both pixel_values and labels
    return {
        "pixel_values": pixel_values,
        "label": examples["label"]
    }

# Apply transforms
print("Applying transforms to datasets...")
train_dataset = train_dataset.with_transform(transforms)
val_dataset = val_dataset.with_transform(transforms)
test_dataset = test_dataset.with_transform(transforms)
print("Transforms applied.")

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,  # Increased batch size
    per_device_eval_batch_size=8,  # Match evaluation batch size
    num_train_epochs=25,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    remove_unused_columns=False,
    save_total_limit=3,  # Keep only the last 3 checkpoints
    # save_strategy='no',  # Disable checkpoint saving
)

# Define metrics
print("Loading metrics...")
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)
print("Metrics loaded.")

# Create trainer
print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=CustomDataCollator(),  # Add this line
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping
)
print("Trainer created.")

# Train model
print("Starting training...")
trainer.train()
print("Training completed.")

# Evaluate model
print("Evaluating model...")
metrics = trainer.evaluate(test_dataset)
print("Evaluation completed.")
print("Metrics:", metrics)

training_loss = trainer.state.log_history

# Extract losses and epochs
loss_values = [entry['loss'] for entry in training_loss if 'loss' in entry]
epochs = [entry['epoch'] for entry in training_loss if 'loss' in entry]

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to unnormalize the image
def unnormalize_image(image_tensor, mean, std):
    image_array = image_tensor.cpu().permute(1, 2, 0).numpy()  # Move tensor to CPU
    image_array = std * image_array + mean
    image_array = np.clip(image_array, 0, 1)
    return image_array

# Set the model to evaluation mode
model.eval()

print("Making predictions...")

# Unnormalize parameters
mean = np.array(feature_extractor.image_mean)
std = np.array(feature_extractor.image_std)

# Prepare a list to store the images and labels
images_and_labels = []

# Iterate over the test dataset
for i, test_example in enumerate(test_dataset):
    print(i, " ", test_example['label'])
    test_image_tensor = test_example['pixel_values'].to(device)
    test_image_array = unnormalize_image(test_image_tensor, mean, std)

    # Make predictions
    with torch.no_grad():
        outputs = model(test_image_tensor.unsqueeze(0).to(device))  # Add batch dimension and move to device
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = id2label[predicted_class_idx]
        true_label = id2label[test_example['label']]

    # Find an example image from the dataset with the predicted label
    predicted_example = next((item for item in test_dataset if item['label'] == predicted_class_idx), None)
    if predicted_example:
        predicted_image_tensor = predicted_example['pixel_values'].to(device)
        predicted_image_array = unnormalize_image(predicted_image_tensor, mean, std)
    else:
        predicted_image_array = np.zeros_like(test_image_array)  # Placeholder if no image is found

    # Store the images and labels
    images_and_labels.append((test_image_array, predicted_image_array, true_label, predicted_label))

# Function to update the plot
def update_plot(index):
    true_image, predicted_image, true_label, predicted_label = images_and_labels[index]
    axes[0].imshow(true_image)
    axes[0].set_title(f"True Label: {true_label}")
    axes[0].axis('off')
    axes[1].imshow(predicted_image)
    axes[1].set_title(f"Predicted Label: {predicted_label}")
    axes[1].axis('off')
    fig.canvas.draw()

# Function to handle key press events
def on_key(event):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(images_and_labels)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(images_and_labels)
    update_plot(current_index)

# Initialize the plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
current_index = 0
update_plot(current_index)

# Connect the key press event to the handler
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()