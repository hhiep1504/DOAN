# Person Re-identification & Feature Classification with YOLOv10 and ResNet50

A comprehensive Python project for detecting individuals in images using YOLOv10 and classifying their demographic features (gender, age, ethnicity) using a fine-tuned ResNet50 model trained on the P-DESTRE dataset.

## Installation

1. **Clone the repository (if available):**
   ```bash
   git clone <your-repository-url>
   cd <your-project-root>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   Create a `requirements.txt` file with the following content:
   ```txt
   torch
   torchvision
   Pillow
   matplotlib
   numpy
   pandas
   tqdm
   opencv-python
   ultralytics
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models:**
   * **YOLOv10:** Download `yolov10n.pt` (or another version if modified in code) from the YOLOv10 repository and place it in the project root directory.
   * **Feature Classifier:** Ensure you have `best_model.pth` (pre-trained model) in the project root. If not available, you need to run the training script.

5. **Prepare data (if needed for training or dataset testing):**
   * Download the P-DESTRE dataset.
   * Extract and place the `P-DESTRE/annotation` folder in the project root.
   * Ensure the `jpg_Extracted_PIDS` folder contains extracted and organized images according to the structure described below.

## Model Training (Optional)

The `train.py` script trains the `FeatureClassifier` model based on a specific annotation file from P-DESTRE.

1. **Configuration:** Open `train.py` and modify the paths (`jpg_dir`, `annotation_file`) if needed. Currently using `P-DESTRE/annotation/08-11-2019-1-1.txt`.
2. **Run training:**
   ```bash
   python train.py
   ```
   The training process will proceed, and the best model (based on validation loss) will be saved to `best_model.pth`.

## Dataset Inspection and Statistics

The `dataset_statistic.py` script helps inspect data structure, load samples, calculate distribution statistics for demographic features (gender, age, ethnicity), and display sample visualizations.

1. **Configuration:** Open `dataset_statistic.py` and ensure the `JPG_DIR` and `ANNOTATION_DIR` paths are correct. You can also modify `NUM_SAMPLES_TO_CHECK` to change the number of samples to inspect.
2. **Run inspection:**
   ```bash
   python dataset_statistic.py
   ```
   The script will print inspection results, statistics, and display sample images (if no errors occur).

## Using the Model (Inference)

There are two main ways to use the trained model:

### 1. Classify features on a single image (assumes image contains only one person and is cropped)

The `infer.py` script loads the `FeatureClassifier` model and runs predictions on a single image file. The image should contain only a face or most of a person's body.

1. **Configuration:** Open `infer.py` and modify the `test_image` variable to point to the image you want to test. Ensure `model_path` points to the correct model file.
2. **Run inference:**
   ```bash
   python infer.py
   ```
   The script will load the model, process the image, make predictions about gender, age, and ethnicity, and display the original image with predictions.

### 2. Detect persons and classify features

The `detect_and_classify.py` script uses YOLOv10 to detect all people in an image, crops each person, and uses `FeatureClassifier` to classify their features.

1. **Configuration:**
   * Open `detect_and_classify.py`.
   * Ensure the paths to YOLO model (`yolov10n.pt`) and `FeatureClassifier` (`best_model.pth`) in the `load_models` function are correct.
   * Modify the `test_image` variable to point to the image you want to process.
2. **Run detection and classification:**
   ```bash
   python detect_and_classify.py
   ```
   The script will:
   * Load both models.
   * Read the input image.
   * Use YOLO to find person bounding boxes.
   * For each detected person:
     * Crop the person's image.
     * Use `FeatureClassifier` to predict features.
     * Print predictions to terminal.
     * Draw bounding boxes and predictions on the original image.
   * Display the final result image.
   * Save the result image to `result.jpg`.

## Other Scripts

* `model.py`: Contains the `FeatureClassifier` class definition using ResNet50 as backbone.
* `dataset_multiannotation_gemini.py`: Defines the `PdestreFeatureDataset` class for reading and processing data from P-DESTRE annotation files and image folders.
* `read_annotations.py`: A utility script to read and display the contents of an annotation file, helping understand the meaning of data columns.

## Project Structure

```
doan/
├── .gitignore              # Files/folders to ignore in Git
├── README.md               # Project overview, installation, and usage
├── requirements.txt        # Python dependencies (pip install -r requirements.txt)
│
├── config/                 # (Optional) Configuration files (YAML, JSON) for paths and model parameters
│   └── config.yaml
│
├── data/
│   ├── raw/                # Raw data, unchanged
│   │   ├── jpg_Extracted_PIDS/
│   │   │   └── ... (Date folders and original images)
│   │   └── P-DESTRE/
│   │       └── annotation/
│   │           └── ... (Original annotation .txt files)
│   │
│   ├── interim/            # Intermediate data after processing steps
│   │   ├── P-DESTRE/
│   │   │   └── annotation_cleaned/
│   │   │       └── ... (Cleaned annotations without duplicates)
│   │   ├── kept_image_paths.txt
│   │   └── final_kept_image_paths.txt
│   │
│   └── processed/          # Final data ready for model
│       └── P-DESTRE/
│           └── annotation_final/
│               └── ... (Final filtered annotations)
│
├── models/                 # Pre-trained or downloaded model weights
│   ├── yolov8n.pt
│   ├── yolov10n.pt
│   └── checkpoints/        # Checkpoints saved during training
│       ├── best_model.pth
│       └── best_weight.pth
│
├── notebooks/              # (Optional) Jupyter Notebooks for exploration and experimentation
│   └── exploratory_analysis.ipynb
│
├── outputs/                # Output results from scripts (images, logs, reports)
│   ├── images/
│   │   ├── result.jpg
│   │   ├── save1.png
│   │   ├── save2.png
│   │   └── test.webp
│   ├── logs/               # Training and inference logs
│   └── reports/            # Reports and tables
│       └── HaHoangHiep_PGNV_DATN.xlsx
│
├── scripts/                # Standalone scripts
│   ├── data_preparation/   # Data processing and preparation
│   │   ├── 01_clean_annotations.py
│   │   ├── 02_filter_similar_images.py
│   │   ├── 03_filter_quality_yolo.py
│   │   ├── 04_create_final_annotations.py
│   │   └── 05_split_dataset.py
│   │
│   ├── analysis/           # Data analysis and statistics scripts
│   │   ├── dataset_statistics.py
│   │   └── read_annotations.py
│   │
│   ├── training/           # Model training scripts
│   │   └── train.py
│   │
│   ├── inference/          # Prediction and classification scripts
│   │   ├── infer.py
│   │   └── detect_and_classify.py
│   │
│   └── visualization/      # Result and data visualization scripts
│       └── show.py
│
└── src/                    # Main source code (reusable classes and functions)
    ├── __init__.py         # Mark as Python package
    ├── datasets.py         # Dataset class definitions (e.g., PdestreFeatureDataset)
    ├── models.py           # Custom model architectures (e.g., feature extractor)
    └── utils.py            # Utility functions for common tasks
```

## Features

- ✅ YOLOv10-based person detection
- ✅ ResNet50-based demographic feature classification
- ✅ Support for P-DESTRE dataset
- ✅ Batch inference and single image processing
- ✅ Data statistics and analysis tools
- ✅ Modular and extensible architecture

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- See `requirements.txt` for complete dependencies

## License

MIT License - See LICENSE file for details

## Authors

- Ha Hoang Hiep

## Contact

For questions or issues, please reach out to the project maintainer.
