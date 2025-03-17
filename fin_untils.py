import os
import random
import json
import torch
import numpy as np
from datasets import Dataset, Features, ClassLabel, Value, load_from_disk
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from typing import Any, Dict, List
from evaluate import load as load_metric
import fingerprint_feature_extractor  # your existing library for minutiae extraction
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F



# ------------------------------------------------
# 3) MINUTIAE FEATURE HELPER FUNCTIONS
# ------------------------------------------------
def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_degrees(x1, y1, x2, y2):
    """
    Returns the angle (in degrees) from (x1,y1) to (x2,y2) in the range [-180, 180].
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def compute_4NN_descriptor(minutia, all_minutiae, k_neighbors=4):
    """
    For a given minutia, find its 4 nearest neighbors (using Euclidean distance)
    among all_minutiae and compute an 8-D descriptor:
      [d1, d2, d3, d4, angle1, angle2, angle3, angle4].
    """
    cx, cy = minutia.locX, minutia.locY

    # Compute distances from this minutia to every other minutia.
    dist_list = []
    for m in all_minutiae:
        if m is minutia:
            continue
        dist = euclidean_distance(cx, cy, m.locX, m.locY)
        dist_list.append((dist, m))

    # Sort by distance and take up to 4 nearest neighbors.
    dist_list.sort(key=lambda x: x[0])
    four_nearest = dist_list[:k_neighbors]

    distances = []
    angles = []
    for dist, neigh in four_nearest:
        nx, ny = neigh.locX, neigh.locY
        ang = angle_degrees(cx, cy, nx, ny)
        distances.append(dist)
        angles.append(ang)

    # If fewer than 4 neighbors, pad with zeros.
    while len(distances) < 4:
        distances.append(0.0)
        angles.append(0.0)

    return distances + angles  # Returns an 8-element list.

def build_retina_descriptor_set(terminations, bifurcations):
    """
    Instead of averaging, build and return the full set of 8-D descriptors (one per minutia)
    for the given terminations and bifurcations.
    If no minutiae are found, return an empty list.
    """
    all_min = terminations + bifurcations
    if len(all_min) == 0:
        return []
    descriptors = []
    for m in all_min:
        desc = compute_4NN_descriptor(m, all_min, k_neighbors=4)
        descriptors.append(desc)
    return descriptors  # A list of 8-D vectors.

def extract_fingerprint_features(example):
    """
    For a given image (provided by its file path in example["path"]),
    load the image, extract minutiae using the provided library, and compute
    the full set of 8-D descriptors. (This function retains the complete set.)
    """
    img_path = example["path"]
    print(f"Processing image: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        return {"fingerprint_features": []}
    # Extract minutiae: terminations and bifurcations.
    terms, bifurs = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False,
        showResult=False, saveResult=False
    )
    print(f"Found {len(terms)} terminations and {len(bifurs)} bifurcations")
    # Compute the full set of 8-D descriptors.
    descriptors = build_retina_descriptor_set(terms, bifurs)

    # added
    # Add normalization
    if descriptors:
        descriptors = np.array(descriptors)
        # Normalize distances
        mean_dist = np.mean(descriptors[:, :4])
        std_dist = np.std(descriptors[:, :4]) + 1e-6
        descriptors[:, :4] = (descriptors[:, :4] - mean_dist) / std_dist
        
        # Normalize angles (circular features)
        descriptors[:, 4:] = np.sin(np.radians(descriptors[:, 4:]))
        
        descriptors = descriptors.tolist()
    
    print(f"Extracted {len(descriptors)} descriptors")
    return {"fingerprint_features": descriptors}

    print(f"Extracted {len(descriptors)} descriptors")
    return {"fingerprint_features": descriptors}

# ------------------------------------------------
# 4) BUILD OR LOAD HF DATASET & COLLATOR FOR SET-BASED DATA
# ------------------------------------------------
# We want to store the full set (variable-length) of descriptors per image.
# We use a custom collator that pads each set to the maximum number in the batch.
@dataclass
class SetCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_descriptors = []
        lengths = []
        labels = []
        # For each sample, convert its list of descriptors (each descriptor is an 8-D list)
        # into a tensor of shape (num_descriptors, 8). If empty, use a single zero vector.
        for f in features:
            desc_list = f["fingerprint_features"]
            if not desc_list:
                desc_tensor = torch.zeros((1, 8), dtype=torch.float32)
                lengths.append(1)
            else:
                desc_tensor = torch.tensor(desc_list, dtype=torch.float32)
                lengths.append(desc_tensor.shape[0])
            batch_descriptors.append(desc_tensor)
            labels.append(f["label"])
        # Pad the list of descriptor tensors so that all samples have the same number of descriptors.
        padded = torch.nn.utils.rnn.pad_sequence(batch_descriptors, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": padded, "lengths": lengths, "labels": labels}
