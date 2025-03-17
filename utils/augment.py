import albumentations as A
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Import the function directly

# # Define an Albumentations augmentation pipeline
# aug_pipeline = A.Compose([
#     A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=1.0),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Rotate(limit=15, p=0.5),
#     A.GaussianBlur(blur_limit=(3, 7), p=0.3),
# ])

# Define augmentation pipeline
aug_pipeline_2 = A.Compose([
    # Geometric augmentations
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, p=0.5),

    # Intensity-based augmentations (preserve vessels)
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),

    # Occlusion augmentation (Cutout for vessel obstruction simulation)
    A.CoarseDropout(max_holes=3, max_height=20, max_width=20, min_holes=1, min_height=10, min_width=10, fill_value=0, p=0.3)
])

def augment_image(image_path, num_augmented=5):
    """Generate a list of augmented images from one image file."""
    # Read the image (Albumentations works with numpy arrays in RGB order)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented_images = []
    for i in range(num_augmented):
        # Apply the augmentation pipeline
        augmented = aug_pipeline_2(image=image)['image']
        augmented_images.append(augmented)
    return augmented_images

def aug_ridb():
    # Example usage: augment all images in the training set and save them in a new folder
    train_images_dir = 'dataset/onDrive-divided-cropped'  # your original train folder
    augmented_dir = 'dataset/onDrive-divided-cropped-augmented'  # your augmented train folder
    os.makedirs(augmented_dir, exist_ok=True)

    # Suppose each class is in a separate subdirectory
    for class_name in os.listdir(train_images_dir):
        class_path = os.path.join(train_images_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        # Create an output directory for this class
        output_class_dir = os.path.join(augmented_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.png')):
                file_path = os.path.join(class_path, filename)
                try:
                    aug_images = augment_image(file_path, num_augmented=5)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
                # Save each augmented image
                base_name, ext = os.path.splitext(filename)
                for idx, aug_img in enumerate(aug_images):
                    # Convert numpy array to PIL Image for saving
                    pil_img = Image.fromarray(aug_img)
                    out_path = os.path.join(output_class_dir, f"{base_name}_aug{idx}{ext}")
                    pil_img.save(out_path)
                    print(f"Saved: {out_path}")


def augment_fire_dataset(source_dir="dataset/FIRE/onDrive-divided", output_dir="dataset/FIRE/onDrive-divided-augmented"):
    """
    Custom augmentation for FIRE dataset with specific requirements:
    - For first image: Create two augmentations (bright+rotated, dark+negative rotated)
    - For second image: Create one augmentation (random brightness and rotation)
    
    Args:
        source_dir: Directory containing person folders with original images
        output_dir: Directory to save augmented images
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Augmenting images in {source_dir} and saving to {output_dir}")
    
    # Define augmentation pipelines
    aug_first_image_1 = A.Compose([
        A.Rotate(limit=(5, 15), p=1.0),   # Random rotation 5-15 degrees
        A.RandomBrightnessContrast(brightness_limit=[0.2, 0.2], contrast_limit=0, p=1.0),  # Brightness 1.2
    ])
    
    aug_first_image_2 = A.Compose([
        A.Rotate(limit=(-25, -5), p=1.0),  # Random rotation -5 to -25 degrees
        A.RandomBrightnessContrast(brightness_limit=[-0.2, -0.2], contrast_limit=0, p=1.0),  # Brightness 0.8
    ])
    
    aug_second_image = A.Compose([
        A.Rotate(limit=(-25, 25), p=1.0),  # Random rotation -25 to 25 degrees
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=0, p=1.0),  # Random brightness 0.8-1.2
    ])
    
    # Process each person folder
    person_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    for person_folder in tqdm(person_folders, desc="Processing person folders"):
        # Create output folder for this person
        person_output_dir = os.path.join(output_dir, person_folder)
        os.makedirs(person_output_dir, exist_ok=True)
        
        # Get images in the person folder
        person_dir = os.path.join(source_dir, person_folder)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < 2:
            print(f"Warning: Folder {person_folder} has fewer than 2 images, skipping...")
            continue
            
        # Sort images to ensure consistent ordering
        images.sort()
        
        # Process first image with two different augmentations
        first_image_path = os.path.join(person_dir, images[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"Error: Could not read {first_image_path}, skipping...")
            continue
            
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        
        # Copy original image to output dir
        base_name, ext = os.path.splitext(images[0])
        Image.fromarray(first_image).save(os.path.join(person_output_dir, images[0]))
        
        # First augmentation: Brighter + positive rotation
        aug_result_1 = aug_first_image_1(image=first_image)['image']
        Image.fromarray(aug_result_1).save(
            os.path.join(person_output_dir, f"{base_name}_bright_pos_rot{ext}")
        )
        
        # Second augmentation: Darker + negative rotation
        aug_result_2 = aug_first_image_2(image=first_image)['image']
        Image.fromarray(aug_result_2).save(
            os.path.join(person_output_dir, f"{base_name}_dark_neg_rot{ext}")
        )
        
        # Process second image with random augmentation
        second_image_path = os.path.join(person_dir, images[1])
        second_image = cv2.imread(second_image_path)
        if second_image is None:
            print(f"Error: Could not read {second_image_path}, skipping...")
            continue
            
        second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
        
        # Copy original image to output dir
        base_name, ext = os.path.splitext(images[1])
        Image.fromarray(second_image).save(os.path.join(person_output_dir, images[1]))
        
        # Random augmentation: Random brightness and rotation
        aug_result_3 = aug_second_image(image=second_image)['image']
        Image.fromarray(aug_result_3).save(
            os.path.join(person_output_dir, f"{base_name}_random_aug{ext}")
        )
        
    print(f"Augmentation complete! Augmented dataset saved to {output_dir}")
    print(f"Created a total of 3 augmented images per person (2 from first image, 1 from second)")

    
if __name__ == "__main__":
    # aug_ridb()
    augment_fire_dataset()

