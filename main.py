import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ----------------------------- #
#         Configuration         #
# ----------------------------- #


# Feature extraction parameters
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
SVM_KERNEL = 'linear'

# Tracking parameters
TRACKER_TYPE = 'KCF'  # Options: 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'

# ----------------------------- #
#         Data Preparation      #
# ----------------------------- #

def load_annotations(annotation_file):
    """
    Load annotations from a CSV file.
    The CSV should have columns: image_filename, x, y, width, height, class_label
    """
    annotations = pd.read_csv(annotation_file)
    return annotations

def load_images(image_dir, annotations):
    """
    Load images and corresponding bounding boxes based on annotations.
    Returns a list of tuples: (image, bounding_box, class_label)
    """
    data = []
    for idx, row in annotations.iterrows():
        image_path = os.path.join(image_dir, row['image_filename'])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image {image_path} could not be loaded.")
            continue
        bbox = [int(row['x']), int(row['y']), int(row['width']), int(row['height'])]
        class_label = row['class_label']
        data.append((image, bbox, class_label))
    return data

def preprocess_image(image, size=(128, 128)):
    """
    Resize and normalize the image. Convert to grayscale.
    """
    resized = cv2.resize(image, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

# ----------------------------- #
#        Feature Extraction     #
# ----------------------------- #

def extract_hog_features(image):
    """
    Extract HOG features from a grayscale image.
    """
    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extract color histogram features from a BGR image.
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features(data):
    """
    Extract features for all images in the dataset.
    Returns feature matrix and labels.
    """
    features = []
    labels = []
    for img, bbox, label in data:
        preprocessed = preprocess_image(img)
        hog_feat = extract_hog_features(preprocessed)
        # Optionally, include color histograms
        # color_feat = extract_color_histogram(img)
        # combined_feat = np.concatenate([hog_feat, color_feat])
        combined_feat = hog_feat  # Using only HOG features for simplicity
        features.append(combined_feat)
        labels.append(label)
    return np.array(features), np.array(labels)

# ----------------------------- #
#      Preparing Training Data #
# ----------------------------- #

def prepare_training_data(features, labels):
    """
    Split the data into training and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    return X_train, X_val, y_train, y_val

# ----------------------------- #
#    Model Selection & Training #
# ----------------------------- #

def train_svm_classifier(X_train, y_train):
    """
    Train an SVM classifier.
    """
    clf = svm.SVC(kernel=SVM_KERNEL, probability=True)
    clf.fit(X_train, y_train)
    return clf

def train_random_forest_classifier(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf

# ----------------------------- #
#         Object Detection      #
# ----------------------------- #

def detect_objects(image, clf, scaler=None):
    """
    Detect objects in an image using the trained classifier.
    Returns predicted bounding boxes and class labels.
    """
    detections = []
    preprocessed = preprocess_image(image)
    hog_feat = extract_hog_features(preprocessed)
    # If using a scaler, apply it here
    # if scaler:
    #     hog_feat = scaler.transform([hog_feat])[0]
    prediction = clf.predict([hog_feat])
    probability = clf.predict_proba([hog_feat])[0]
    # Threshold probability as needed
    detections.append({
        'bbox': [0, 0, preprocessed.shape[1], preprocessed.shape[0]],  # Placeholder
        'class_label': prediction[0],
        'confidence': probability[np.argmax(probability)]
    })
    return detections

# ----------------------------- #
#         Object Tracking       #
# ----------------------------- #

def initialize_tracker(tracker_type, initial_bbox, frame):
    """
    Initialize the tracker with the first frame and bounding box.
    """
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")
    
    tracker.init(frame, tuple(initial_bbox))
    return tracker

def track_objects(tracker, frame):
    """
    Update tracker with the new frame and return the updated bounding box.
    """
    success, bbox = tracker.update(frame)
    if success:
        return bbox
    else:
        return None

# ----------------------------- #
#       Evaluation Metrics     #
# ----------------------------- #

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in the format [x, y, width, height]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(clf, X_val, y_val):
    """
    Evaluate the classifier using classification metrics.
    """
    y_pred = clf.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

# ----------------------------- #
#          Main Pipeline        #
# ----------------------------- #

def main():
    # Step 1: Load Annotations and Images
    annotations = load_annotations(ANNOTATION_FILE)
    data = load_images(IMAGE_DIR, annotations)
    print(f"Loaded {len(data)} images with annotations.")

    # Step 2: Feature Extraction
    features, labels = extract_features(data)
    print(f"Extracted features with shape: {features.shape}")

    # Step 3: Prepare Training Data
    X_train, X_val, y_train, y_val = prepare_training_data(features, labels)
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Step 4: Train Classifier
    clf = train_svm_classifier(X_train, y_train)
    # Alternatively, use Random Forest
    # clf = train_random_forest_classifier(X_train, y_train)
    print("Classifier trained.")

    # Step 5: Evaluate Model
    evaluate_model(clf, X_val, y_val)

    # Step 6: Object Detection on New Images
    new_image_path = 'path_to_new_image.jpg'  # Replace with actual image path
    new_image = cv2.imread(new_image_path)
    if new_image is not None:
        detections = detect_objects(new_image, clf)
        for det in detections:
            bbox = det['bbox']
            label = det['class_label']
            confidence = det['confidence']
            cv2.rectangle(new_image, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 2)
            cv2.putText(new_image, f"{label} {confidence:.2f}",
                        (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image with detections
        cv2.imshow("Detections", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Failed to load new image from {new_image_path}.")

    # Step 7: Object Tracking (Example with Video)
    video_path = 'path_to_video.mp4'  # Replace with actual video path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}.")
        return

    # Assume first frame contains the object to track
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame from the video.")
        return

    # For demonstration, use the first detection as the initial bounding box
    initial_bbox = [50, 50, 100, 100]  # Replace with actual bbox
    tracker = initialize_tracker(TRACKER_TYPE, initial_bbox, first_frame)
    print("Tracker initialized.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = track_objects(tracker, frame)
        if bbox:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frame, "Tracking", (p1[0], p1[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        else:
            cv2.putText(frame, "Lost", (50,80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0,0,255),2)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
