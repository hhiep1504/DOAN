import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from ultralytics import YOLO

class DatasetCreator:
    def __init__(self, output_dir="pedestrian_dataset"):
        """
        Initialize the dataset creator
        
        Args:
            output_dir: Directory to save the processed dataset
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.annotations = []
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize pedestrian detector
        self.detector = self.load_detector()
        
    def load_detector(self):
        """Load a pre-trained pedestrian detector model"""
        # We'll use YOLOv10 for pedestrian detection
        try:
            # Load YOLOv10n model using the ultralytics library
            model = YOLO('yolov10n.pt') 
            # Filtering for classes will be done during the prediction call
            return model
        except Exception as e:
            print(f"Error loading YOLOv10 detector: {e}")
            print("Please ensure you have an internet connection and the 'ultralytics' package installed.")
            print("You might need to install or update it: pip install ultralytics")
            return None
    
    def extract_frames(self, video_path, frame_interval=30):
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            frame_interval: Number of frames to skip (e.g., 30 means 1 fps for 30fps video)
            
        Returns:
            List of extracted frames
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Extracting approximately {total_frames/frame_interval} frames")
        
        count = 0
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                count += 1
                pbar.update(1)
                
        cap.release()
        return frames
    
    def detect_pedestrians(self, frames):
        """
        Detect pedestrians in frames
        
        Args:
            frames: List of frames
            
        Returns:
            List of pedestrian bounding boxes for each frame
        """
        if self.detector is None:
            print("Detector not loaded. Skipping detection.")
            return []
        
        all_pedestrians = []
        
        print("Detecting pedestrians...")
        for i, frame_np in enumerate(tqdm(frames)):
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_np)
            
            # Run detection using the ultralytics YOLO model
            # Filter for persons (class 0) and confidence > 0.5
            # verbose=False to reduce console output from YOLO
            results_list = self.detector(pil_image, classes=0, conf=0.5, verbose=False)
            
            if not results_list:
                all_pedestrians.append([])
                continue

            results = results_list[0] # Get the Results object for the single image
            
            frame_pedestrians = []
            # results.boxes contains detected boxes with properties like xyxy, conf, cls
            for box_data in results.boxes:
                # box_data.xyxy is a tensor of shape [1, 4] for each detected box
                # It contains [xmin, ymin, xmax, ymax] in pixel coordinates
                x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                
                # Extract pedestrian image from the original numpy frame
                pedestrian_img = frame_np[y1:y2, x1:x2]
                frame_pedestrians.append({
                    'bbox': (x1, y1, x2, y2),
                    'image': pedestrian_img
                })
            
            all_pedestrians.append(frame_pedestrians)
        
        return all_pedestrians
    
    def save_pedestrian(self, pedestrian_img, idx):
        """
        Save a pedestrian image
        
        Args:
            pedestrian_img: Image of the pedestrian
            idx: Index for the file name
            
        Returns:
            Path to the saved image
        """
        if pedestrian_img.size == 0:
            return None
            
        # Resize to consistent dimensions
        try:
            pedestrian_img = cv2.resize(pedestrian_img, (100, 300))
        except Exception as e:
            print(f"Error resizing image: {e}")
            return None
            
        # Save the image
        image_path = os.path.join(self.images_dir, f"{idx:05d}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(pedestrian_img, cv2.COLOR_RGB2BGR))
        
        return f"{idx:05d}.jpg"
    
    def annotate_interactively(self, image_path):
        """
        Interactively annotate a pedestrian image
        
        Args:
            image_path: Path to the pedestrian image
            
        Returns:
            Dictionary with annotations
        """
        # Load and display the image
        img = cv2.imread(os.path.join(self.images_dir, image_path))
        cv2.imshow("Pedestrian", img)
        
        print(f"\nAnnotating {image_path}")
        
        # Get annotations
        gender = input("Gender (0:Male, 1:Female, 2:Unknown): ")
        age = input("Age (0:Child, 1:Teen, 2:YoungAdult, 3:Adult, 4:MiddleAge, 5:Senior, 6:Elderly, 7:Young, 8:Unknown): ")
        ethnicity = input("Ethnicity (0:Asian, 1:Black, 2:Caucasian, 3:Other, 4:Unknown): ")
        beard = input("Beard (0:Yes, 1:No, 2:Unknown): ")
        glasses = input("Glasses (0:Regular, 1:Sun, 2:No, 3:Unknown): ")
        
        print("Accessories (enter comma-separated values):")
        print("0:Hat/Cap, 1:Backpack, 2:Bag/Purse, 3:Headphones, 4:Mask, 5:Scarf, 6:Jewelry, 7:None/Unknown")
        accessories = input("Accessories: ")
        
        cv2.destroyAllWindows()
        
        return {
            'image_id': image_path,
            'gender': int(gender) if gender.isdigit() else 2,
            'age': int(age) if age.isdigit() else 8,
            'ethnicity': int(ethnicity) if ethnicity.isdigit() else 4,
            'beard': int(beard) if beard.isdigit() else 2,
            'glasses': int(glasses) if glasses.isdigit() else 3,
            'accessories': accessories if accessories else "7"
        }
    
    def process_video(self, video_path, interactive=False):
        """
        Process a video to extract pedestrians and annotate them
        
        Args:
            video_path: Path to the video file
            interactive: Whether to annotate pedestrians interactively
            
        Returns:
            None
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Detect pedestrians
        all_pedestrians = self.detect_pedestrians(frames)
        
        # Process each pedestrian
        idx = len(self.annotations)
        for frame_idx, frame_pedestrians in enumerate(all_pedestrians):
            for ped in frame_pedestrians:
                pedestrian_img = ped['image']
                image_path = self.save_pedestrian(pedestrian_img, idx)
                
                if image_path is None:
                    continue
                
                if interactive:
                    annotation = self.annotate_interactively(image_path)
                else:
                    # Default annotations (to be filled later)
                    annotation = {
                        'image_id': image_path,
                        'gender': 2,  # Unknown
                        'age': 8,     # Unknown
                        'ethnicity': 4, # Unknown
                        'beard': 2,   # Unknown
                        'glasses': 3, # Unknown
                        'accessories': "7" # None/Unknown
                    }
                
                self.annotations.append(annotation)
                idx += 1
        
        # Save annotations
        self.save_annotations()
        
    def save_annotations(self):
        """Save annotations to CSV file"""
        df = pd.DataFrame(self.annotations)
        df.to_csv(os.path.join(self.output_dir, "annotations.csv"), index=False)
        print(f"Saved {len(self.annotations)} annotations")
    
    def load_annotations(self):
        """Load existing annotations"""
        csv_path = os.path.join(self.output_dir, "annotations.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.annotations = df.to_dict('records')
            print(f"Loaded {len(self.annotations)} existing annotations")
    
    def process_videos(self, video_dir, interactive=False):
        """
        Process all videos in a directory
        
        Args:
            video_dir: Directory containing video files
            interactive: Whether to annotate pedestrians interactively
        """
        # Load existing annotations if any
        self.load_annotations()
        
        # Get all video files
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            self.process_video(video_path, interactive)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create pedestrian attribute dataset from videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="pedestrian_dataset", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Annotate pedestrians interactively")
    
    args = parser.parse_args()
    
    creator = DatasetCreator(output_dir=args.output_dir)
    creator.process_videos(args.video_dir, interactive=args.interactive)

if __name__ == "__main__":
    main()