from pathlib import Path
from PIL import Image
import imagehash
import os
from tqdm import tqdm
from collections import defaultdict

def filter_similar_images_in_person_folder(person_dir, hash_size=8, hamming_threshold=5):
    """
    Filters similar images within a single person's directory using pHash.

    Args:
        person_dir (Path): Path to the directory containing images for one person.
        hash_size (int): The size of the hash (higher means more detail, but slower).
                         Must be a power of 2. 8 is a common default.
        hamming_threshold (int): Maximum Hamming distance to consider images similar.
                                Lower values mean images must be *more* similar to be grouped.
                                0 means identical hashes. 5 is a reasonable starting point.

    Returns:
        list[Path]: A list of Paths to the images deemed unique enough to keep.
    """
    images_to_process = sorted(list(person_dir.glob('*.jpg')))
    if not images_to_process:
        return []

    hashes = {} # Store hash -> list of file paths with that exact hash
    image_paths = {} # Store hash -> path of the first image encountered with that hash

    # --- Calculate hashes ---
    for img_path in images_to_process:
        try:
            img = Image.open(img_path)
            # Ensure hash_size is appropriate if needed, phash uses 8x8 blocks by default
            # perceptual hash works best on images of reasonable size
            current_hash = imagehash.phash(img, hash_size=hash_size)

            if current_hash not in hashes:
                 hashes[current_hash] = []
                 image_paths[current_hash] = img_path # Keep track of the first image for this hash
            hashes[current_hash].append(img_path)

        except Exception as e:
            print(f"\nWarning: Could not process image {img_path.name} in {person_dir.name}. Error: {e}")
            continue # Skip broken images

    # --- Group similar hashes ---
    unique_hashes = list(hashes.keys())
    kept_image_paths = set() # Use a set to automatically handle adding the same image multiple times
    processed_hashes = set()

    # Sort hashes to ensure deterministic output (optional but good practice)
    # Sorting imagehash objects directly might not be straightforward,
    # converting to string for sorting is an option if needed.
    # For now, processing order might vary slightly run-to-run for similar hashes.

    for h1_idx, h1 in enumerate(unique_hashes):
        if h1 in processed_hashes:
            continue

        # Keep the first image encountered for this hash group
        # We use image_paths[h1] which stores the path of the *first* image seen with hash h1
        representative_path = image_paths[h1]
        kept_image_paths.add(representative_path)
        processed_hashes.add(h1)

        # Find other hashes similar to h1
        for h2_idx in range(h1_idx + 1, len(unique_hashes)):
            h2 = unique_hashes[h2_idx]
            if h2 in processed_hashes:
                continue

            distance = h1 - h2 # Calculate Hamming distance
            if distance <= hamming_threshold:
                # h2 is considered similar to h1, mark h2 as processed
                # We DON'T add images from h2's group because h1's representative covers them
                processed_hashes.add(h2)

    return list(kept_image_paths)


def process_dataset_for_duplicates(jpg_root_dir, output_file="kept_image_paths.txt", hash_size=8, hamming_threshold=5):
    """
    Processes the entire dataset structure to find and filter similar images per person.

    Args:
        jpg_root_dir (str or Path): Path to the root directory (e.g., 'jpg_Extracted_PIDS').
        output_file (str): File path to save the list of images to keep.
        hash_size (int): Hash size for imagehash.phash.
        hamming_threshold (int): Hamming distance threshold for similarity.
    """
    root_path = Path(jpg_root_dir)
    all_kept_paths = []
    total_original_images = 0
    date_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    print(f"Found {len(date_dirs)} date directories in {root_path}")

    for date_dir in tqdm(date_dirs, desc="Processing Dates"):
        person_dirs = [p for p in date_dir.iterdir() if p.is_dir()]
        for person_dir in tqdm(person_dirs, desc=f"Processing Persons in {date_dir.name}", leave=False):
            original_count = len(list(person_dir.glob('*.jpg')))
            total_original_images += original_count

            kept_paths_for_person = filter_similar_images_in_person_folder(
                person_dir,
                hash_size=hash_size,
                hamming_threshold=hamming_threshold
            )
            all_kept_paths.extend(kept_paths_for_person)

    # --- Save the results ---
    print(f"\n--- Filtering Summary ---")
    print(f"Total original images found: {total_original_images}")
    print(f"Total images kept after filtering: {len(all_kept_paths)}")
    removed_count = total_original_images - len(all_kept_paths)
    if total_original_images > 0:
        removed_percent = (removed_count / total_original_images) * 100
        print(f"Removed {removed_count} similar images ({removed_percent:.2f}% reduction).")
    else:
        print("No images found to process.")

    # Sort paths for consistent output file
    all_kept_paths.sort()

    with open(output_file, 'w') as f:
        for img_path in all_kept_paths:
            # Save relative path from the root directory for easier use
            try:
                 relative_path = img_path.relative_to(root_path)
                 f.write(f"{relative_path}\n")
            except ValueError:
                 # Handle cases where the path might not be relative (e.g., symlinks, edge cases)
                 f.write(f"{img_path}\n") # Save absolute path as fallback

    print(f"\nList of kept image paths saved to: {Path(output_file).resolve()}")


# --- Cấu hình ---
JPG_ROOT_DIR = 'jpg_Extracted_PIDS' # Đường dẫn gốc chứa các thư mục ngày
OUTPUT_FILE = 'kept_image_paths.txt' # File lưu danh sách ảnh giữ lại
HASH_SIZE = 8         # Kích thước hash (nên là lũy thừa của 2, 8 là mặc định tốt)
HAMMING_THRESHOLD = 5 # Ngưỡng khoảng cách Hamming (0-5 thường là hợp lý, bạn có thể thử nghiệm)

# --- Chạy quá trình lọc ---
if __name__ == "__main__":
    process_dataset_for_duplicates(
        JPG_ROOT_DIR,
        output_file=OUTPUT_FILE,
        hash_size=HASH_SIZE,
        hamming_threshold=HAMMING_THRESHOLD
    )