import os
import shutil

# Source directory containing the images
source_dir = 'dataset/GRATINA/onDrive'

# Destination directory where folders will be created
dest_dir = 'dataset/GRATINA/onDrive-divided'

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    if not filename.endswith('_bin_seg.png'):
        # Split the filename to extract the person number
        parts = filename.split('_')
        if len(parts) >= 1:
            person_number = parts[0]
            person_number_2 = parts[1]
            # Create a folder for the person if it doesn't exist
            person_folder = os.path.join(dest_dir, f'person_{person_number}')
            os.makedirs(person_folder, exist_ok=True)
            # Copy the file to the person’s folder
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(person_folder, filename)
            shutil.copy(src_file, dst_file)

# import os
# import shutil

# # Source directory containing the images
# source_dir = 'dataset/images'

# # Destination directory where folders will be created
# dest_dir = 'dataset/images-divided'

# # Ensure the destination directory exists
# os.makedirs(dest_dir, exist_ok=True)

# # Iterate over all files in the source directory
# for filename in os.listdir(source_dir):
#     # Split the filename to extract the person number
#     parts = filename.split('_')
#     if len(parts) >= 2:
#         person_number = parts[1]
#         # Create a folder for the person if it doesn't exist
#         person_folder = os.path.join(dest_dir, f'person_{person_number}')
#         os.makedirs(person_folder, exist_ok=True)
#         # Copy the file to the person’s folder
#         src_file = os.path.join(source_dir, filename)
#         dst_file = os.path.join(person_folder, filename)
#         shutil.copy(src_file, dst_file)