import json

def delete_person_annotations_in_frame_range(
    annotation_file_path,
    person_id_to_delete,
    start_frame,
    end_frame,
    output_file_path=None
):
    """
    Deletes annotations for a specific person_id within a given frame range
    from an annotations.json file.

    Args:
        annotation_file_path (str): Path to the input annotations.json file.
        person_id_to_delete (int): The track_id of the person to delete.
        start_frame (int): The starting frame index (inclusive).
        end_frame (int): The ending frame index (inclusive).
        output_file_path (str, optional): Path to save the modified annotations.
            If None, the original file will be overwritten. Defaults to None.
    """
    try:
        with open(annotation_file_path, 'r') as f:
            annotations_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{annotation_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{annotation_file_path}'")
        return

    modified_count = 0
    frames_affected = set()

    # Iterate through a copy of keys because we might modify the dictionary
    for frame_idx_str in list(annotations_data.keys()):
        try:
            frame_idx_int = int(frame_idx_str)
        except ValueError:
            print(f"Warning: Skipping non-integer frame key '{frame_idx_str}'")
            continue

        if start_frame <= frame_idx_int <= end_frame:
            persons_in_frame = annotations_data[frame_idx_str]
            
            # Filter out the specified person_id
            updated_persons_in_frame = [
                person_data for person_data in persons_in_frame
                if not (person_data.get("track_id") == person_id_to_delete)
            ]

            if len(updated_persons_in_frame) < len(persons_in_frame):
                modified_count += (len(persons_in_frame) - len(updated_persons_in_frame))
                frames_affected.add(frame_idx_int)
                
                # If all persons in this frame were the one to be deleted and the list becomes empty,
                # you might choose to remove the frame key itself or keep it as an empty list.
                # This script keeps it as an empty list if all were deleted.
                # If you want to remove the frame key if it becomes empty:
                # if not updated_persons_in_frame:
                #     del annotations_data[frame_idx_str]
                # else:
                #     annotations_data[frame_idx_str] = updated_persons_in_frame
                annotations_data[frame_idx_str] = updated_persons_in_frame


    if modified_count > 0:
        print(f"Removed {modified_count} annotation(s) for person_id {person_id_to_delete}.")
        print(f"Frames affected in the range [{start_frame}-{end_frame}]: {sorted(list(frames_affected))}")

        # Determine output path
        if output_file_path is None:
            output_file_path = annotation_file_path # Overwrite original
            print(f"Overwriting original file: '{output_file_path}'")
        else:
            print(f"Saving modified data to: '{output_file_path}'")

        try:
            with open(output_file_path, 'w') as f:
                json.dump(annotations_data, f, indent=4) # Use indent for readability
            print("Successfully saved the modified annotations.")
        except IOError:
            print(f"Error: Could not write to file '{output_file_path}'")
    else:
        print(f"No annotations found for person_id {person_id_to_delete} in the frame range [{start_frame}-{end_frame}]. No changes made.")


if __name__ == "__main__":
    # --- Configuration ---
    annotation_file = r'C:\Users\ADMIN\Documents\doan\outputs\hiep_dataset\phone_5\annotations.json'  # Path to your annotations file
    person_id_to_remove = 1
    start_frame_index = 0  # Inclusive
    end_frame_index = 99    # Inclusive

    # To save to a new file instead of overwriting, set a path here:
    # output_file = "annotations_modified.json"
    output_file = None # Set to None to overwrite the original file

    # --- Confirmation ---
    action = "overwrite the original file" if output_file is None else f"save to '{output_file}'"
    confirm = input(
        f"This will attempt to remove person_id {person_id_to_remove} from frames "
        f"{start_frame_index} to {end_frame_index} in '{annotation_file}' "
        f"and {action}.\nAre you sure you want to proceed? (y/n): "
    )

    if confirm.lower() == 'y':
        print("Proceeding with modification...")
        delete_person_annotations_in_frame_range(
            annotation_file,
            person_id_to_remove,
            start_frame_index,
            end_frame_index,
            output_file
        )
    else:
        print("Operation cancelled by user.")