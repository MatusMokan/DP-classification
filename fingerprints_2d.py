import os
import random
import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
# ---------- Helper Functions ----------

# skript 15 dobrych osob 5 utocnikov

random.seed(42)
np.random.seed(42)

def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_degrees(x1, y1, x2, y2):
    """Compute the angle in degrees from (x1,y1) to (x2,y2) in [-180,180]."""
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))

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

def build_retina_template(feature_points, k=4):
    """
    Build a template as an R x 8 matrix.
    For each feature point, find its k nearest neighbors and compute:
      [d1, d2, d3, d4, theta1, theta2, theta3, theta4].
    """
    if len(feature_points) == 0:
        return np.empty((0, 8), dtype=np.float32)
    R = len(feature_points)
    template = np.zeros((R, 8), dtype=np.float32)
    for i, (cx, cy) in enumerate(feature_points):
        dist_list = []
        for j, (nx, ny) in enumerate(feature_points):
            if i == j:
                continue
            dist = euclidean_distance(cx, cy, nx, ny)
            dist_list.append((dist, nx, ny))
        dist_list.sort(key=lambda x: x[0])
        nearest = dist_list[:k]
        dist_vals, angle_vals = [], []
        for (dist, nx, ny) in nearest:
            dist_vals.append(dist)
            angle_vals.append(angle_degrees(cx, cy, nx, ny))
        # Zero-pad if fewer than k neighbors
        while len(dist_vals) < k:
            dist_vals.append(0.0)
            angle_vals.append(0.0)
        template[i, :k] = dist_vals
        template[i, k:] = angle_vals
    return template

def extract_template(image_path):
    """
    Load image, binarize with Otsu, skeletonize, extract feature points,
    and build a retina template.
    Returns the template (R x 8 numpy array).
    """
    vessel_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if vessel_gray is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    _, vessel_bin = cv2.threshold(vessel_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = (vessel_bin > 0).astype(np.uint8)
    skeleton = skeletonize(bin_img).astype(np.uint8)
    feature_points = crossing_number_features(skeleton)
    template = build_retina_template(feature_points, k=4)
    return template

def compute_min_distance(template1, template2):
    """
    Compute the minimum L2 distance between any pair of rows from two templates.
    """
    if template1.shape[0] == 0 or template2.shape[0] == 0:
        return float("inf")
    dists = []
    for row1 in template1:
        for row2 in template2:
            dists.append(np.linalg.norm(row1 - row2))
    return np.min(dists)

def match_live_template(live_template, database, threshold=5):
    """
    Simple matching: for a given live_template, compute the minimal distance to every template in the database.
    Returns the best matching subject ID and the best (minimum) distance.
    If the best distance is above the threshold, the test image is rejected.
    """
    best_subject = None
    best_distance = float("inf")
    for subject_id, templates in database.items():
        for stored_template in templates:
            dist = compute_min_distance(live_template, stored_template)
            if dist < best_distance:
                best_distance = dist
                best_subject = subject_id
    if best_distance <= threshold:
        return best_subject, best_distance
    else:
        return None, best_distance

def match_live_template_TSM(live_template, database, distance_threshold=5, vote_threshold=15):
    """
    Enhanced Matching Strategy (TSM):
    For each feature vector in the live_template, find the subject whose stored template row 
    yields the minimal L2 distance (if below the distance_threshold) and vote for that subject.
    If the maximum vote meets or exceeds vote_threshold, return that subject.
    Otherwise, return None.
    """
    votes = {subject_id: 0 for subject_id in database.keys()}
    for live_vec in live_template:
        best_sub = None
        best_dist = float("inf")
        for subject_id, templates in database.items():
            for stored_template in templates:
                for stored_vec in stored_template:
                    dist = np.linalg.norm(live_vec - stored_vec)
                    if dist < best_dist:
                        best_dist = dist
                        best_sub = subject_id
        if best_dist <= distance_threshold and best_sub is not None:
            votes[best_sub] += 1
    best_subject = max(votes, key=votes.get)
    if votes[best_subject] >= vote_threshold:
        return best_subject, votes[best_subject]
    else:
        return None, votes[best_subject]

def plot_distance_distribution(genuine_dists, imposter_dists):
    """Plot histograms of minimum distances for genuine and imposter matches."""
    plt.figure(figsize=(8, 6))
    plt.hist(genuine_dists, bins=20, alpha=0.7, label='Genuine', color='green')
    plt.hist(imposter_dists, bins=20, alpha=0.7, label='Imposter', color='red')
    plt.xlabel("Minimum L2 Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distance Distribution for Genuine vs Imposter Matches")
    plt.show()

def compute_distance_distribution(test_set, database):
    """
    For each test image, compute the minimum distance (using simple matching)
    between its template and the database.
    Returns two lists: one for genuine matches and one for imposter matches.
    """
    genuine_dists = []
    imposter_dists = []
    for true_subject, img_path, is_genuine in tqdm(test_set, desc="Computing distance distribution"):
        live_template = extract_template(img_path)
        _, best_dist = match_live_template(live_template, database, threshold=1e9)
        if is_genuine:
            genuine_dists.append(best_dist)
        else:
            imposter_dists.append(best_dist)
    return genuine_dists, imposter_dists


def plot_far_frr_vs_threshold_both(mode, test_set, database, thresholds=np.linspace(2, 5, 10), vote_threshold=15):
    """
    Vyhodnotí a vykreslí FAR a FRR vs. threshold pre daný matching režim.
    
    Parameters:
      mode (str): 'simple' pre jednoduché matching, 'tsm' pre TSM matching.
      test_set (list): Zoznam testovacích vzoriek, každý prvok je (subject, img_path, is_genuine).
      database (dict): Databáza (enrollment), kľúč = subject, hodnota = zoznam template (každý reduced PCA template).
      thresholds (np.array): Pole threshold hodnôt, cez ktoré sa iteruje.
      vote_threshold (int): Počet hlasov potrebných pri TSM matching (iba ak je mode=='tsm').
    """
    FAR_list = []
    FRR_list = []
    total_genuine = sum(1 for _, _, genuine in test_set if genuine)
    total_attackers = sum(1 for _, _, genuine in test_set if not genuine)

    print(f"\n--- FAR & FRR vs. Threshold ({mode.capitalize()} Matching) ---")
    for th in tqdm(thresholds, desc="Threshold Loop"):
        print(f"Threshold: {th:.2f}")
        true_accepts = 0
        false_rejects = 0
        false_accepts = 0
        true_rejects = 0
        
        for true_subject, img_path, is_genuine in tqdm(test_set, desc="Matching", leave=False):
            live_template = extract_template(img_path)
            if mode == "simple":
                matched_subject, dist = match_live_template(live_template, database, threshold=th)
            elif mode == "tsm":
                matched_subject, votes = match_live_template_TSM(live_template, database, distance_threshold=th, vote_threshold=vote_threshold)
                dist = votes  # pre debug vypis môžeme použiť počet hlasov
            else:
                raise ValueError("Mode must be 'simple' or 'tsm'")
                
            if is_genuine:
                if matched_subject == true_subject:
                    true_accepts += 1
                else:
                    false_rejects += 1
            else:
                print(f"[DEBUG] Attacker image {true_subject} -> threshold {th:.2f}: matched with {matched_subject} (value = {dist:.2f})")
                if matched_subject is None:
                    true_rejects += 1
                else:
                    false_accepts += 1
        
        # Vytvorenie konfúznej matice pre aktuálny threshold
        confusion_mat = np.array([[true_accepts, false_rejects],
                                    [false_accepts, true_rejects]])
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Genuine', 'Imposter'], yticklabels=['Genuine', 'Imposter'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix at Threshold {th:.2f} ({mode.capitalize()} Matching)')
        plt.show()

        FAR = false_accepts / total_attackers if total_attackers > 0 else 0
        FRR = false_rejects / total_genuine if total_genuine > 0 else 0
        FAR_list.append(FAR)
        FRR_list.append(FRR)

    # Výpočet aproximatívneho EER
    abs_diffs = [abs(FAR_list[i] - FRR_list[i]) for i in range(len(thresholds))]
    min_index = np.argmin(abs_diffs)
    EER_threshold = thresholds[min_index]
    EER = (FAR_list[min_index] + FRR_list[min_index]) / 2.0

    print(f"\nApproximate EER Threshold ({mode.capitalize()} Matching): {EER_threshold:.2f}")
    print(f"Approximate EER ({mode.capitalize()} Matching): {EER*100:.2f}%")

    # Vykreslenie FAR a FRR vs. threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, FAR_list, label="FAR", marker="o", color="red")
    plt.plot(thresholds, FRR_list, label="FRR", marker="o", color="blue")
    plt.axvline(x=EER_threshold, color='gray', linestyle='--', label="EER Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title(f"FAR & FRR vs. Threshold ({mode.capitalize()} Matching)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"FAR_FRR_vs_Threshold_{mode}.png")
    # plt.show()  # Re-enable showing the plot

# ---------- Database Saving/Loading ----------

DATABASE_FILE = "database.pkl"
# DATABASE_FILE = "database4D.pkl"


def save_database(db, filename=DATABASE_FILE):
    with open(filename, "wb") as f:
        pickle.dump(db, f)

def load_database(filename=DATABASE_FILE):
    with open(filename, "rb") as f:
        return pickle.load(f)


# ---------- Main Pipeline ----------

if __name__ == "__main__":
    base_dir = "dataset/onDrive-divided-cropped"
    subjects = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    enrolled_subjects = subjects[:15]  # first 15 persons (genuine subjects)
    attacker_subjects = subjects[15:]   # last 5 persons (attackers)

    # Load or build enrollment database
    if os.path.exists(DATABASE_FILE):
        print("Loading existing enrollment database...")
        database = load_database(DATABASE_FILE)
    else:
        database = {}
        print("Building enrollment database...")
        for subject in tqdm(enrolled_subjects, desc="Enrolling subjects"):
            subject_folder = os.path.join(base_dir, subject)
            images = sorted(os.listdir(subject_folder))
            enroll_images = images[:3]  # first 3 images for enrollment
            for img_name in tqdm(enroll_images, desc=f"Processing {subject}", leave=False):
                img_path = os.path.join(subject_folder, img_name)
                template = extract_template(img_path)
                if subject not in database:
                    database[subject] = []
                database[subject].append(template)
        print(f"Database built with {len(database)} subjects and {sum(len(v) for v in database.values())} enrolled images.")
        save_database(database)
        print("Enrollment database saved.")

    # Build Test Set
    test_set = []
    print("Building test set...")
    for subject in tqdm(enrolled_subjects, desc="Test set (genuine)"):
        subject_folder = os.path.join(base_dir, subject)
        images = sorted(os.listdir(subject_folder))
        test_images = images[3:5]  # remaining 2 images
        for img_name in test_images:
            img_path = os.path.join(subject_folder, img_name)
            test_set.append((subject, img_path, True))
    for subject in tqdm(attacker_subjects, desc="Test set (attackers)"):
        subject_folder = os.path.join(base_dir, subject)
        images = sorted(os.listdir(subject_folder))
        chosen = random.sample(images, 2)
        for img_name in chosen:
            img_path = os.path.join(subject_folder, img_name)
            test_set.append((subject, img_path, False))
    print(f"Total test images: {len(test_set)}")

    # Compute Distance Distribution for Threshold Tuning
    genuine_dists, imposter_dists = compute_distance_distribution(test_set, database)
    print(f"Average Genuine Distance: {np.mean(genuine_dists):.2f}")
    print(f"Average Imposter Distance: {np.mean(imposter_dists):.2f}")
    plot_distance_distribution(genuine_dists, imposter_dists)

    # ----- Plot FAR & FRR vs. Threshold for Simple Matching -----
    # Pre jednoduché matching:
    # plot_far_frr_vs_threshold_both("simple", test_set, database)

    # Pre TSM matching s vote_threshold = 5:
    # plot_far_frr_vs_threshold_both("tsm", test_set, database, vote_threshold=5)

    # --- Matching and Evaluation (Simple Matching and TSM Matching) ---
    true_accepts_simple = 0
    false_rejects_simple = 0
    false_accepts_simple = 0
    true_rejects_simple = 0

    true_accepts_tsm = 0
    false_rejects_tsm = 0
    false_accepts_tsm = 0
    true_rejects_tsm = 0

    total_genuine = sum(1 for _, _, genuine in test_set if genuine)
    total_attackers = sum(1 for _, _, genuine in test_set if not genuine)

    # Set thresholds for final evaluation (adjust these based on your plots)
    distance_threshold = 3.8  # example threshold for simple matching
    vote_threshold = 5     # example vote threshold for TSM matching
    print(f"\nUsing distance threshold: {distance_threshold}")
    print(f"Using vote threshold: {vote_threshold}")
    print("\n--- Matching and Evaluation ---")

    print("Matching test images (Simple Matching)...")
    for true_subject, img_path, is_genuine in tqdm(test_set, desc="Simple Matching"):
        live_template = extract_template(img_path)
        matched_subject, _ = match_live_template(live_template, database, threshold=distance_threshold)
        if is_genuine:
            if matched_subject == true_subject:
                true_accepts_simple += 1
            else:
                false_rejects_simple += 1
        else:
            if matched_subject is None:
                true_rejects_simple += 1
            else:
                false_accepts_simple += 1

    print("Matching test images (TSM Matching)...")
    for true_subject, img_path, is_genuine in tqdm(test_set, desc="TSM Matching"):
        live_template = extract_template(img_path)
        matched_subject, votes = match_live_template_TSM(live_template, database, distance_threshold=distance_threshold, vote_threshold=vote_threshold)
        if is_genuine:
            if matched_subject == true_subject:
                true_accepts_tsm += 1
            else:
                false_rejects_tsm += 1
        else:
            print(f"[DEBUG] Attacker image {true_subject} votes = {votes} -> matched with {matched_subject}")
            if matched_subject is None:
                true_rejects_tsm += 1
            else:
                false_accepts_tsm += 1

    accuracy_simple = true_accepts_simple / total_genuine if total_genuine > 0 else 0
    FAR_simple = false_accepts_simple / total_attackers if total_attackers > 0 else 0
    FRR_simple = false_rejects_simple / total_genuine if total_genuine > 0 else 0

    accuracy_tsm = true_accepts_tsm / total_genuine if total_genuine > 0 else 0
    FAR_tsm = false_accepts_tsm / total_attackers if total_attackers > 0 else 0
    FRR_tsm = false_rejects_tsm / total_genuine if total_genuine > 0 else 0

    print("\n--- Evaluation (Simple Matching) ---")
    print(f"True Accepts: {true_accepts_simple} out of {total_genuine}")
    print(f"False Rejects: {false_rejects_simple} out of {total_genuine}")
    print(f"False Accepts: {false_accepts_simple} out of {total_attackers}")
    print(f"True Rejects: {true_rejects_simple} out of {total_attackers}")
    print(f"Accuracy on genuine images: {accuracy_simple*100:.2f}%")
    print(f"False Acceptance Rate (FAR): {FAR_simple*100:.2f}%")
    print(f"False Rejection Rate (FRR): {FRR_simple*100:.2f}%")

    print("\n--- Evaluation (TSM Matching) ---")
    print(f"True Accepts: {true_accepts_tsm} out of {total_genuine}")
    print(f"False Rejects: {false_rejects_tsm} out of {total_genuine}")
    print(f"False Accepts: {false_accepts_tsm} out of {total_attackers}")
    print(f"True Rejects: {true_rejects_tsm} out of {total_attackers}")
    print(f"Accuracy on genuine images: {accuracy_tsm*100:.2f}%")
    print(f"False Acceptance Rate (FAR): {FAR_tsm*100:.2f}%")
    print(f"False Rejection Rate (FRR): {FRR_tsm*100:.2f}%")

    