## Object Detection and Tracking Pipeline

This Python project provides a comprehensive pipeline for object detection and tracking using HOG (Histogram of Oriented Gradients) features, machine learning classifiers, and OpenCV tracking algorithms. It supports feature extraction, classification, object detection, and tracking in both images and videos.

---

## Features

1. **Data Preparation:**
   - Loads annotations from a CSV file.
   - Reads images from a specified directory.
   - Preprocesses images (resizing, normalization, and grayscale conversion).

2. **Feature Extraction:**
   - Extracts HOG features for image classification.
   - (Optional) Supports color histogram features for enhanced detection.

3. **Machine Learning:**
   - Provides training pipelines for:
     - Support Vector Machine (SVM) classifier.
     - Random Forest classifier.
   - Evaluates the model using classification metrics such as precision, recall, and confusion matrix.

4. **Object Detection:**
   - Detects objects in new images using the trained classifier.
   - Visualizes detections with bounding boxes and labels.

5. **Object Tracking:**
   - Tracks objects in video sequences using OpenCV tracking algorithms (e.g., KCF, CSRT).
   - Supports initialization with detected bounding boxes.

6. **Evaluation Metrics:**
   - Calculates Intersection over Union (IoU) for bounding box evaluation.

---

## Installation

### Requirements

Install the following Python libraries before running the code:

- `numpy`
- `pandas`
- `opencv-python`
- `scikit-image`
- `scikit-learn`
- `matplotlib`

You can install these dependencies using:

```bash
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib
```

---

## Usage

### 1. **Configuration**
Set the paths to your image directory and annotation file in the script:

```python
IMAGE_DIR = 'path_to_images'
ANNOTATION_FILE = 'path_to_annotations.csv'
```

Configure feature extraction and model parameters as needed.

### 2. **Running the Pipeline**
Run the script directly:

```bash
python main.py
```

The pipeline performs the following steps:
- Loads data and annotations.
- Extracts features using HOG.
- Trains a classifier (SVM or Random Forest).
- Evaluates the classifier on a validation set.
- Detects objects in new images.
- Tracks objects in a video sequence.

### 3. **File Format**

#### Annotations CSV
The annotation file should have the following format:

| image_filename | x   | y   | width | height | class_label |
|----------------|-----|-----|-------|--------|-------------|
| image1.jpg     | 50  | 30  | 100   | 120    | cat         |
| image2.jpg     | 70  | 50  | 80    | 90     | dog         |

---

## Outputs

- **Classification Metrics:** A detailed classification report and confusion matrix.
- **Object Detection Visualizations:** Bounding boxes and labels overlaid on images.
- **Object Tracking:** Video frames with updated bounding boxes for tracked objects.

---

## Customization

1. **Change Features:**
   Uncomment and modify `extract_color_histogram()` to include color histograms with HOG features.

2. **Add New Trackers:**
   Update `TRACKER_TYPE` in the configuration to switch between available OpenCV trackers.

3. **Use Different Classifiers:**
   Switch between SVM and Random Forest in the `main()` function.

---

## Limitations

- The pipeline currently uses only a single detection in tracking initialization.
- HOG-based feature extraction may not perform well on highly complex datasets. Consider using deep learning-based feature extractors for better results.

---

## License

This project is open-source and available under the MIT License. Contributions are welcome!
