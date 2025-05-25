import os

def count_images_in_nested_dataset(dataset_path):
    """
    Counts image files within a nested dataset structure.

    The structure is assumed to be:
    dataset_path/
        MVI_video_folder_1/
            bbox_image/
                person_folder_1/
                    image1.jpg
                    image2.png
                    ...
                person_folder_2/
                    image3.jpg
                    ...
            ... (other files/folders in MVI_video_folder_1)
        MVI_video_folder_2/
            bbox_image/
                person_folder_3/
                    imageA.jpg
                    ...
            ...
        ...

    Args:
        dataset_path (str): The path to the main dataset directory (e.g., the one containing "MVI_9421").

    Returns:
        int: The total count of image files found.
    """
    total_image_count = 0
    # Common image file extensions (add more if needed)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

    if not os.path.isdir(dataset_path):
        print(f"Error: The base directory '{os.path.abspath(dataset_path)}' does not exist.")
        return 0

    #print(f"Scanning for images in: {os.path.abspath(dataset_path)}\n")

    # 1. Iterate through video folders (e.g., MVI_9421)
    for video_folder_name in os.listdir(dataset_path):
        video_folder_path = os.path.join(dataset_path, video_folder_name)

        if os.path.isdir(video_folder_path):
            # print(f"Processing video folder: {video_folder_name}")

            # 2. Check for 'bbox_image' subfolder
            bbox_image_folder_path = os.path.join(video_folder_path, "bbox_image")

            if os.path.isdir(bbox_image_folder_path):
                # print(f"  Found 'bbox_image' in {video_folder_name}")

                # 3. Iterate through person folders inside 'bbox_image'
                for person_folder_name in os.listdir(bbox_image_folder_path):
                    person_folder_path = os.path.join(bbox_image_folder_path, person_folder_name)

                    if os.path.isdir(person_folder_path):
                        # print(f"    Processing person folder: {person_folder_name} (in {video_folder_name})")
                        images_in_this_person_folder = 0
                        
                        # 4. Count images in the person folder
                        for image_file_name in os.listdir(person_folder_path):
                            image_file_path = os.path.join(person_folder_path, image_file_name)

                            if os.path.isfile(image_file_path) and \
                               image_file_name.lower().endswith(image_extensions):
                                total_image_count += 1
                                images_in_this_person_folder += 1
                        
                        #if images_in_this_person_folder > 0:
                            #print(f"  Found {images_in_this_person_folder} images in: '{video_folder_name}/bbox_image/{person_folder_name}'")
                        # else:
                        #     print(f"  No images found in: '{video_folder_name}/bbox_image/{person_folder_name}'")
            # else:
            #     print(f"  No 'bbox_image' folder found in {video_folder_name}")
        # else:
            # print(f"Skipping non-directory entry at video folder level: {video_folder_name}")


    return total_image_count

if __name__ == "__main__":
    dataset_parent_path = "outputs/hiep_dataset"  

    total_images = count_images_in_nested_dataset(dataset_parent_path)

    if total_images > -1:
        print(f"Total number of images found inside '{dataset_parent_path}': {total_images}") 
  