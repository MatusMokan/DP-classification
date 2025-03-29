import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold



# Define paths
# DATASET_PATH = "dataset/GRATINA100/onDrive-divided-augmented-more"
DATASET_PATH = "dataset/onDrive-divided-cropped-augmented"
MODEL_PATH = "models/retina_svm"

# Make sure the model path exists
os.makedirs(MODEL_PATH, exist_ok=True)

from sklearn.cluster import KMeans

def build_codebook(all_templates, k=32):
    """
    Create a KMeans codebook (visual vocabulary) from all template vectors.
    Input:
        all_templates: list of (R x 8) matrices from training images
        k: number of clusters (visual words)
    Output:
        Trained KMeans model
    """
    all_vectors = np.vstack(all_templates)
    print(f"Clustering {all_vectors.shape[0]} feature vectors into {k} visual words...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_vectors)
    return kmeans

def build_bof_histogram(template, kmeans_model, k=32):
    """
    Create a BoF histogram for a single template using a trained codebook.
    Input:
        template: R x 8 matrix (feature vectors for one image)
        kmeans_model: Trained KMeans model
        k: number of clusters (should match model)
    Output:
        histogram: 1D vector of length k
    """
    if template.shape[0] == 0:
        return np.zeros(k)
    labels = kmeans_model.predict(template)
    hist, _ = np.histogram(labels, bins=np.arange(k+1))

    return hist.astype(np.float32) / np.sum(hist)  # Normalize histogram


# We'll use existing functions from your code
def crossing_number_features(skeleton):
    """
    Return a list of (x, y) feature points from a skeletonized image
    using the crossing-number technique.
    """
    rows, cols = skeleton.shape
    feature_pts = []
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0,  1),
                         (1,  1), (1,  0), (1, -1),
                         (0, -1)]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if skeleton[y, x] == 1:
                cn_sum = 0
                for i in range(8):
                    curr_val = skeleton[y + neighbors_offsets[i][0],
                                        x + neighbors_offsets[i][1]]
                    next_val = skeleton[y + neighbors_offsets[(i + 1) % 8][0],
                                        x + neighbors_offsets[(i + 1) % 8][1]]
                    cn_sum += abs(int(curr_val) - int(next_val))
                CN = cn_sum / 2.0
                # Consider a point as a feature if CN==1 (endpoint), 3 (bifurcation) or >3 (crossing)
                if CN == 1 or CN == 3 or CN > 3:
                    feature_pts.append((x, y))
    return feature_pts

def angle_degrees(x1, y1, x2, y2):
    """Compute the angle in degrees from (x1,y1) to (x2,y2) in [-180,180]."""
    dx = x2 - x1
    dy = y2 - y1
    return np.degrees(np.arctan2(dy, dx))

def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def build_retina_template_invariant(feature_points, k=4):
    """
    Build a rotation-invariant template as an R x 8 matrix.
    """
    if len(feature_points) == 0:
        return np.empty((0, 8), dtype=np.float32)
    
    R = len(feature_points)
    template = np.zeros((R, 8), dtype=np.float32)
    
    for i, (cx, cy) in enumerate(feature_points):
        # Compute all angles from this point to every other point
        angles = []
        for j, (nx, ny) in enumerate(feature_points):
            if i == j:
                continue
            angles.append(angle_degrees(cx, cy, nx, ny))
        
        base_angle = np.median(angles) if angles else 0
        
        # Now compute distances and relative angles
        dist_rel_list = []
        for j, (nx, ny) in enumerate(feature_points):
            if i == j:
                continue
            dist = euclidean_distance(cx, cy, nx, ny)
            abs_angle = angle_degrees(cx, cy, nx, ny)
            rel_angle = (abs_angle - base_angle) % 360
            dist_rel_list.append((dist, rel_angle))
        
        # Sort by distance and pick the k nearest neighbors
        dist_rel_list.sort(key=lambda x: x[0])
        nearest = dist_rel_list[:k]
        
        # Unpack distances and angles, and zero-pad if needed
        dist_vals, angle_vals = zip(*nearest) if nearest else ([], [])
        dist_vals = list(dist_vals)
        angle_vals = list(angle_vals)
        while len(dist_vals) < k:
            dist_vals.append(0.0)
            angle_vals.append(0.0)
            
        template[i, :k] = dist_vals
        template[i, k:] = angle_vals
    
    return template

def extract_template(image_path):
    vessel_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if vessel_gray is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    _, vessel_bin = cv2.threshold(vessel_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = (vessel_bin > 0).astype(np.uint8)
    skeleton = skeletonize(bin_img).astype(np.uint8)
    feature_points = crossing_number_features(skeleton)
    template = build_retina_template_invariant(feature_points, k=4)
    return template

# def prepare_dataset():
#     """Load data, extract templates, and prepare for SVM training"""
#     X_train = []
#     y_train = []
#     X_test = []
#     y_test = []
    
#     # List all subject folders
#     subject_folders = [d for d in sorted(os.listdir(DATASET_PATH)) 
#                       if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
#     print(f"Found {len(subject_folders)} subject folders")
    
#     for subject in tqdm(subject_folders, desc="Processing subjects"):
#         subject_path = os.path.join(DATASET_PATH, subject)
#         images = sorted(os.listdir(subject_path))
        
#         if len(images) < 2:
#             print(f"Warning: Subject {subject} has less than 2 images, skipping")
#             continue
        
#         # Use first image for training, second for testing
#         train_img = os.path.join(subject_path, images[0])
#         test_img = os.path.join(subject_path, images[1])
        
#         try:
#             # Extract templates
#             train_template = extract_template(train_img)
#             test_template = extract_template(test_img)
            
#             if len(train_template) == 0 or len(test_template) == 0:
#                 print(f"Warning: Empty template for subject {subject}, skipping")
#                 continue
                
#             # Create feature vector by averaging all rows in the template
#             train_features = np.mean(train_template, axis=0)
#             test_features = np.mean(test_template, axis=0)
            
#             # Add to dataset
#             X_train.append(train_features)
#             y_train.append(subject)
#             X_test.append(test_features)
#             y_test.append(subject)
            
#         except Exception as e:
#             print(f"Error processing subject {subject}: {e}")
    
#     return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def prepare_dataset_bof(kmeans_model, k=32):
    X_train, y_train, X_test, y_test = [], [], [], []

    subject_folders = [d for d in sorted(os.listdir(DATASET_PATH)) 
                      if os.path.isdir(os.path.join(DATASET_PATH, d))]

    for subject in tqdm(subject_folders, desc="Building BoF dataset"):
        subject_path = os.path.join(DATASET_PATH, subject)
        images = sorted(os.listdir(subject_path))

        if len(images) >= 5:
            train_imgs = images[:3]
            test_imgs = images[3:5]
        elif len(images) >= 2:
            train_imgs = [images[0]]
            test_imgs = [images[1]]
        else:
            continue  # skip if not enough images

        try:
            # Process training images
            for img_name in train_imgs:
                img_path = os.path.join(subject_path, img_name)
                template = extract_template(img_path)
                if template.shape[0] == 0:
                    continue
                hist = build_bof_histogram(template, kmeans_model, k)
                X_train.append(hist)
                y_train.append(subject)

            # Process testing images
            for img_name in test_imgs:
                img_path = os.path.join(subject_path, img_name)
                template = extract_template(img_path)
                if template.shape[0] == 0:
                    continue
                hist = build_bof_histogram(template, kmeans_model, k)
                X_test.append(hist)
                y_test.append(subject)

        except Exception as e:
            print(f"[WARNING] {subject}: {e}")

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def train_and_evaluate_svm():
    print("Preparing visual words...")

    subject_folders = [d for d in sorted(os.listdir(DATASET_PATH)) 
                       if os.path.isdir(os.path.join(DATASET_PATH, d))]

    all_template_rows = []
    for subject in tqdm(subject_folders, desc="Collecting features for codebook"):
        subject_path = os.path.join(DATASET_PATH, subject)
        images = sorted(os.listdir(subject_path))
        if len(images) < 2:
            continue
        img_path = os.path.join(subject_path, images[0])
        template = extract_template(img_path)
        if template.shape[0] > 0:
            all_template_rows.append(template)

    k = 32
    kmeans_model = build_codebook(all_template_rows, k=k)

    # Prepare BoF dataset
    X_train, y_train, X_test, y_test = prepare_dataset_bof(kmeans_model, k=k)

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_scaled, y_train)

    # Save
    with open(os.path.join(MODEL_PATH, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm, f)
    with open(os.path.join(MODEL_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, "confusion_matrix.png"))

    # Probability histogram
    probas = svm.predict_proba(X_test_scaled)
    max_probas = np.max(probas, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_probas, bins=20)
    plt.title('Distribution of Maximum Class Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(MODEL_PATH, "probability_distribution.png"))

    return svm, scaler, X_test_scaled, y_test

def extract_template_features(template):
    """
    Extract richer features from a retina template.
    For each of the 8 columns in the template, compute:
      - mean, standard deviation, minimum, and maximum
    Output:
      A 1D feature vector of length 32 (8 columns x 4 statistics)
    """
    if template.shape[0] == 0:
        return np.zeros(32)  # Return zero vector if template is empty
    
    features = []
    for col in range(template.shape[1]):
        col_data = template[:, col]
        features.append(np.mean(col_data))
        features.append(np.std(col_data))
        features.append(np.min(col_data))
        features.append(np.max(col_data))
    return np.array(features)

def prepare_dataset_richer_features():
    X_train, y_train, X_test, y_test = [], [], [], []
    subject_folders = [d for d in sorted(os.listdir(DATASET_PATH))
                       if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    for subject in tqdm(subject_folders, desc="Processing subjects"):
        subject_path = os.path.join(DATASET_PATH, subject)
        images = sorted(os.listdir(subject_path))
        
        # Skip subject if there are fewer than 3 images
        if len(images) < 3:
            continue
        
        # Use the first 3 images for training and the remaining for testing
        train_imgs = images[:20]
        test_imgs = images[20:]
        
        try:
            # Process training images
            for train_img in train_imgs:
                train_img_path = os.path.join(subject_path, train_img)
                template = extract_template(train_img_path)
                if template.shape[0] == 0:
                    continue
                features = extract_template_features(template)
                X_train.append(features)
                y_train.append(subject)
            
            # Process testing images
            for test_img in test_imgs:
                test_img_path = os.path.join(subject_path, test_img)
                template = extract_template(test_img_path)
                if template.shape[0] == 0:
                    continue
                features = extract_template_features(template)
                X_test.append(features)
                y_test.append(subject)
        except Exception as e:
            print(f"Error processing {subject}: {e}")
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    print("Preparing richer feature dataset...")
    X_train, y_train, X_test, y_test = prepare_dataset_richer_features()
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # print("Training SVM classifier...")
    # svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    # svm.fit(X_train_scaled, y_train)
    
    # y_pred = svm.predict(X_test_scaled)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))



    
# Use 3-fold cross-validation since each class has around 3 training samples.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define a pipeline with StandardScaler and SVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

# Define parameter grid for SVM
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.001],
    'svm__kernel': ['rbf', 'poly']
}

# Initialize GridSearchCV with the adjusted cross-validation splitter
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)

print("Performing grid search...")
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get the best model and evaluate on the test set
best_model = grid_search.best_estimator_
X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
y_pred = best_model.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     np.random.seed(42)
#     random.seed(42)
    
#     # Train and evaluate
#     svm, scaler, X_test, y_test = train_and_evaluate_svm()
    
#     print("\nTesting with different confidence thresholds:")
#     thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
#     for threshold in thresholds:
#         # Get probabilities
#         probas = svm.predict_proba(X_test)
#         max_proba_indices = np.argmax(probas, axis=1)
#         max_probas = np.max(probas, axis=1)
        
#         # Apply threshold
#         y_pred = []
#         for i, max_prob in enumerate(max_probas):
#             if max_prob >= threshold:
#                 y_pred.append(svm.classes_[max_proba_indices[i]])
#             else:
#                 y_pred.append("unknown")
        
#         # Calculate accuracy (ignoring unknowns)
#         matches = sum(1 for p, t in zip(y_pred, y_test) if p != "unknown" and p == t)
#         total_classified = sum(1 for p in y_pred if p != "unknown")
        
#         if total_classified > 0:
#             precision = matches / total_classified
#         else:
#             precision = 0
            
#         recall = matches / len(y_test)
        
#         print(f"\nThreshold {threshold:.1f}:")
#         print(f"  Classified: {total_classified}/{len(y_test)} ({total_classified/len(y_test)*100:.1f}%)")
#         print(f"  Precision: {precision:.4f}")
#         print(f"  Recall: {recall:.4f}")