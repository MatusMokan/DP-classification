import fingerprint_feature_extractor
import cv2
import sys
from icecream import ic
import numpy as np  
from typing import Any, Dict, List


# Print the Python path for debugging
print("Python path:", sys.path)


# img = cv2.imread('dataset/onDrive/IM000001_1_bin_seg.png', 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
# img = cv2.imread('dataset/onDrive/IM000001_1_seg.png', 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
img = cv2.imread('dataset/onDrive-divided-cropped-augmented/person_1/IM000001_1_seg_aug0.png', 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library

FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)

ic(FeaturesTerminations)
ic(FeaturesBifurcations)

feature = FeaturesTerminations[0]
feature2 = FeaturesBifurcations[0]

# Print __dict__ to see which attributes are available
ic(feature.__dict__)
ic(feature2.__dict__)

# Access the correct attributes you see there
ic(feature.locX, feature.locY, feature.Orientation, feature.Type)

# Convert them to a dictionary
extracted = {
    "x": feature.locX,
    "y": feature.locY,
    "orientation": feature.Orientation,
    "type": feature.Type
}

ic(extracted)
ic(len(FeaturesTerminations))
ic(len(FeaturesBifurcations))


def aggregate_minutiae_features(terminations: List[Any], bifurcations: List[Any]) -> Dict[str, Any]:
    """
    Aggregates minutiae features into a fixed-length vector using histograms
    and statistical summaries for Termination and Bifurcation features.

    :param terminations: List of Termination minutiae features.
    :param bifurcations: List of Bifurcation minutiae features.
    :return: Aggregated feature vector.
    """

    def compute_features(features, name_prefix):
        # Extract locX, locY, and Orientation
        locX = [f.locX for f in features]
        locY = [f.locY for f in features]
        orientations = [o for f in features for o in f.Orientation]

        # Histograms (10 bins each for locX, locY, and Orientation)
        locX_hist = np.histogram(locX, bins=10, range=(0, 512))[0]
        locY_hist = np.histogram(locY, bins=10, range=(0, 512))[0]
        orientation_hist = np.histogram(orientations, bins=10, range=(-180, 180))[0]

        # Normalize histograms
        locX_hist = locX_hist / np.sum(locX_hist) if np.sum(locX_hist) > 0 else locX_hist
        locY_hist = locY_hist / np.sum(locY_hist) if np.sum(locY_hist) > 0 else locY_hist
        orientation_hist = orientation_hist / np.sum(orientation_hist) if np.sum(orientation_hist) > 0 else orientation_hist

        # Statistical summaries
        locX_stats = [np.mean(locX), np.std(locX), np.min(locX), np.max(locX)] if locX else [0, 0, 0, 0]
        locY_stats = [np.mean(locY), np.std(locY), np.min(locY), np.max(locY)] if locY else [0, 0, 0, 0]
        orientation_stats = [np.mean(orientations), np.std(orientations), np.min(orientations), np.max(orientations)] if orientations else [0, 0, 0, 0]

        # Combine into a single vector
        ic(locX_hist, locY_hist, orientation_hist, locX_stats, locY_stats, orientation_stats)
        combined = np.concatenate([locX_hist, locY_hist, orientation_hist, locX_stats, locY_stats, orientation_stats])

        return {f"{name_prefix}_features": combined.tolist()}

    # Compute features for terminations and bifurcations
    term_features = compute_features(terminations, "termination")
    bifur_features = compute_features(bifurcations, "bifurcation")

    # Combine termination and bifurcation features into a single vector
    aggregated_features = np.concatenate([term_features["termination_features"], bifur_features["bifurcation_features"]])

    return {"fingerprint_features": aggregated_features.tolist()}


# Example usage with your extracted minutiae
aggregated = aggregate_minutiae_features(FeaturesTerminations, FeaturesBifurcations)

print("Aggregated Features Length:", len(aggregated["fingerprint_features"]))
print("Aggregated Features Example:", aggregated["fingerprint_features"][:10])  # Print first 10 features

ic(len(aggregated["fingerprint_features"]))