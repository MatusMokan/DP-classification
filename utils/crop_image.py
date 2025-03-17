import cv2
import numpy as np
import os

def crop_white_region(image_path, margin=20):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image is loaded correctly
    if image is None:
        raise ValueError(f"Image not loaded: {image_path}")
    
    # Get image dimensions
    height, width = image.shape

    # Find all white pixel positions (assuming white is 255)
    white_pixels = np.column_stack(np.where(image > 1))  # (y, x) coordinates

    if white_pixels.size == 0:
        raise ValueError(f"No white pixels detected in {image_path}")

    # Find the first and last white pixels based on y-coordinate
    first_white_y, first_white_x = white_pixels[0]
    last_white_y, last_white_x = white_pixels[-1]

    # Look for white pixels in a range around y=500 (e.g., y=480 to y=520)
    y_range_min, y_range_max = 480, 520
    white_pixels_y_500 = white_pixels[(white_pixels[:, 0] >= y_range_min) & (white_pixels[:, 0] <= y_range_max)]

    if white_pixels_y_500.size > 0:
        first_white_x = white_pixels_y_500[:, 1].min()  # Leftmost white pixel near y=500
        last_white_x = white_pixels_y_500[:, 1].max()  # Rightmost white pixel near y=500

    # Define cropping bounds with margin
    x_min = max(first_white_x - margin, 0)
    x_max = min(last_white_x + margin, width)

    # Calculate width with margin
    cropped_width = x_max - x_min

    # Adjust height to match width (centered around first and last white pixel y positions)
    center_y = (first_white_y + last_white_y) // 2
    half_size = cropped_width // 2

    y_min = center_y - half_size
    y_max = center_y + half_size

    # If y_min or y_max is out of bounds, add black padding
    if y_min < 0 or y_max > height:
        pad_top = abs(y_min) if y_min < 0 else 0
        pad_bottom = (y_max - height) if y_max > height else 0

        # Compute final bounds inside image
        y_min = max(y_min, 0)
        y_max = min(y_max, height)

        # Crop the valid part of the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Add black padding to make it square
        cropped_image = cv2.copyMakeBorder(
            cropped_image,
            top=pad_top, bottom=pad_bottom, left=0, right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=0  # Black padding
        )
    else:
        # Normal cropping
        cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

# Define source and output folders
input_folder = "dataset/onDrive-divided"
output_folder = "dataset/onDrive-divided-cropped"

# Walk through every folder and file in the input folder
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(".png"):
            input_path = os.path.join(root, filename)
            # Create a corresponding output directory preserving relative folder structure
            rel_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Crop the image
                cropped = crop_white_region(input_path)
                # Save the resulting cropped image
                cv2.imwrite(output_path, cropped)
                print(f"Cropped image saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")