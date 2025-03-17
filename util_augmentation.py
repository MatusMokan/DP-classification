import os
import random
from PIL import Image, ImageEnhance

# Adjust these paths as needed
SRC_ROOT = "dataset/onDrive-divided-cropped"
DST_TRAIN = "second/dataset/train_data"
DST_VAL = "second/dataset/val_data"
DST_TEST = "second/dataset/test_data"

# Number of images to pick for each class
NUM_TRAIN = 1
NUM_VAL = 2
NUM_TEST = 2

# Brightness factors to apply for the 2 brightness-adjusted copies
BRIGHTNESS_FACTORS = [0.8, 1.2]  # 80% (darker) and 120% (lighter)

# Rotation angles (in degrees)
#  5 times clockwise: 5, 10, 15, 20, 25
#  5 times counterclockwise: -5, -10, -15, -20, -25
ROTATION_ANGLES = [5, 10, 15, 20, 25, -5, -10, -15, -20, -25]

def ensure_dir_exists(folder_path):
    """Create the folder if it doesn't already exist."""
    os.makedirs(folder_path, exist_ok=True)

def save_image(img, save_path):
    """Helper to save a PIL image to disk."""
    img.save(save_path)

def adjust_brightness(img, factor):
    """Return a brightness-adjusted copy of img."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def rotate_image(img, angle):
    """Return a rotated copy of img (with black fill)."""
    return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))

def main():
    # 1) Gather class names (subfolders in SRC_ROOT)
    class_names = [
        d for d in os.listdir(SRC_ROOT) 
        if os.path.isdir(os.path.join(SRC_ROOT, d))
    ]
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # 2) For each class, gather images, shuffle, pick subsets
    for class_name in class_names:
        class_folder = os.path.join(SRC_ROOT, class_name)
        images = [
            img for img in os.listdir(class_folder) 
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        # We will assume each class has at least 5 images
        if len(images) < 5:
            print(f"Skipping class '{class_name}' because it has fewer than 5 images.")
            continue
        
        # Shuffle for randomness
        random.shuffle(images)
        
        train_imgs = images[:NUM_TRAIN]  # 3
        val_img = images[NUM_TRAIN:NUM_TRAIN+NUM_VAL]  # 1
        test_img = images[NUM_TRAIN+NUM_VAL:NUM_TRAIN+NUM_VAL+NUM_TEST]  # 1
        
        # 3) Create output folders for this class
        train_out_dir = os.path.join(DST_TRAIN, class_name)
        val_out_dir = os.path.join(DST_VAL, class_name)
        test_out_dir = os.path.join(DST_TEST, class_name)
        
        ensure_dir_exists(train_out_dir)
        ensure_dir_exists(val_out_dir)
        ensure_dir_exists(test_out_dir)
        
        # 4) Process training images
        for idx, img_name in enumerate(train_imgs):
            src_path = os.path.join(class_folder, img_name)
            with Image.open(src_path) as img:
                # Convert to RGB to avoid issues (e.g., grayscale or RGBA)
                img = img.convert("RGB")
                
                # Save the original
                base_name = os.path.splitext(img_name)[0]
                original_out = os.path.join(train_out_dir, f"{base_name}_orig.jpg")
                save_image(img, original_out)
                
                # 2 brightness-adjusted copies
                for bf in BRIGHTNESS_FACTORS:
                    bright_img = adjust_brightness(img, bf)
                    brightness_tag = f"bright{bf}"
                    bright_out = os.path.join(
                        train_out_dir, f"{base_name}_{brightness_tag}.jpg"
                    )
                    save_image(bright_img, bright_out)
                
                # 10 rotated copies (5 clockwise, 5 counterclockwise)
                for angle in ROTATION_ANGLES:
                    rot_img = rotate_image(img, angle)
                    angle_tag = f"rot{angle}"
                    rot_out = os.path.join(
                        train_out_dir, f"{base_name}_{angle_tag}.jpg"
                    )
                    save_image(rot_img, rot_out)
        
        # 5) Process validation image
        for val_name in val_img:
            src_path = os.path.join(class_folder, val_name)
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                val_out_path = os.path.join(val_out_dir, val_name)
                save_image(img, val_out_path)
        
        # 6) Process test image
        for test_name in test_img:
            src_path = os.path.join(class_folder, test_name)
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                test_out_path = os.path.join(test_out_dir, test_name)
                save_image(img, test_out_path)
                
        print(f"Class '{class_name}' done. Train: {len(train_imgs)} base images (each => 1 orig + 2 brightness + 10 rotates), "
              f"Val: {len(val_img)}, Test: {len(test_img)}")
    
    print("Data augmentation and splitting completed.")

if __name__ == "__main__":
    main()