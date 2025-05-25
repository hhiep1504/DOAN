""" Hiển thị một lưới các ảnh từ một danh sách đường dẫn được cung cấp trong file text.
Chức năng:
- Đọc danh sách các đường dẫn ảnh tương đối từ một file text đầu vào.
- Tìm các file ảnh tương ứng dựa trên một thư mục gốc được cấu hình.
- Mở và hiển thị một số lượng giới hạn (hoặc tất cả) các ảnh tìm thấy được
  trong một cửa sổ Matplotlib dưới dạng lưới.
- Hữu ích để kiểm tra trực quan kết quả của các bước lọc hoặc chuẩn bị dữ liệu.
"""

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import math
import os

# --- Configuration ---

BASE_IMAGE_DIR = Path("data/raw/jpg_Extracted_PIDS") # <--- CHANGE THIS PATH!!!
INPUT_FILE_PATH = Path("final_sampled.txt") # <--- CHANGE THIS FILENAME/PATH!!!

# You can limit the number of images shown at once if the list is very long
MAX_IMAGES_TO_SHOW = 100 # Set a limit, or use float('inf') to show all from file

# --- Read Image Paths from File ---
relative_image_paths_from_file = []
if not INPUT_FILE_PATH.exists() or not INPUT_FILE_PATH.is_file():
    print(f"Error: Input file '{INPUT_FILE_PATH}' not found or is not a file.")
    exit()

print(f"Reading image paths from: '{INPUT_FILE_PATH}'")
try:
    with open(INPUT_FILE_PATH, 'r') as f:
        for line in f:
            # Remove leading/trailing whitespace (like newline chars)
            cleaned_line = line.strip()
            # Ignore empty lines and lines starting with # (comments)
            if cleaned_line and not cleaned_line.startswith('#'):
                # Replace backslashes with forward slashes for consistency with pathlib
                path_with_forward_slashes = cleaned_line.replace('\\', '/')
                relative_image_paths_from_file.append(path_with_forward_slashes)
    print(f"Read {len(relative_image_paths_from_file)} paths from the file.")
except Exception as e:
    print(f"Error reading input file '{INPUT_FILE_PATH}': {e}")
    exit()

if not relative_image_paths_from_file:
    print("Input file was empty or contained no valid paths.")
    exit()

# --- Main Script Logic (Mostly unchanged) ---
if not BASE_IMAGE_DIR.exists() or not BASE_IMAGE_DIR.is_dir():
    print(f"Error: Base directory '{BASE_IMAGE_DIR}' does not exist or is not a directory.")
    print("Please set the BASE_IMAGE_DIR variable correctly.")
    exit()

images_to_display = []
titles = []
print(f"\nSearching for up to {min(MAX_IMAGES_TO_SHOW, len(relative_image_paths_from_file))} images specified in the file, relative to '{BASE_IMAGE_DIR}'...")

loaded_count = 0
# Use the list read from the file now
for i, rel_path_str in enumerate(relative_image_paths_from_file):
    if loaded_count >= MAX_IMAGES_TO_SHOW:
        print(f"Reached display limit of {MAX_IMAGES_TO_SHOW} images.")
        break

    # Construct full path using pathlib
    full_path = BASE_IMAGE_DIR / rel_path_str

    if full_path.exists() and full_path.is_file():
        try:
            img = Image.open(full_path)
            images_to_display.append(img.copy())
            titles.append(rel_path_str) # Use relative path as title
            loaded_count += 1
            img.close() # Close the file handle after copying
        except UnidentifiedImageError:
             print(f"Warning: Pillow could not identify image file: '{full_path}'. Skipping.")
        except Exception as e:
            print(f"Warning: Could not open image '{full_path}'. Error: {e}. Skipping.")
    else:
        print(f"Warning: Image not found or is not a file at '{full_path}'. Skipping.")

# --- Display (Unchanged) ---
num_images = len(images_to_display)
if num_images == 0:
    print("\nNo valid images found or loaded from the specified paths.")
else:
    print(f"\nDisplaying {num_images} loaded images...")
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    fig_width = cols * 3
    fig_height = rows * 3.5
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle(f"Displaying {num_images} Images from '{INPUT_FILE_PATH.name}'", fontsize=16)

    if num_images == 1:
        ax_list = [axes]
    else:
        ax_list = axes.flatten()

    for i in range(num_images):
        ax_list[i].imshow(images_to_display[i])
        ax_list[i].set_title(titles[i], fontsize=7)
        ax_list[i].axis('off')

    for j in range(num_images, len(ax_list)):
        ax_list[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect slightly
    plt.show()

print("\nScript finished.")
del images_to_display