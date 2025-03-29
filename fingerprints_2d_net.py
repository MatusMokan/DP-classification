import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage.morphology import skeletonize
from tqdm import tqdm
import random

# ------------- CONFIG -------------
DATASET_PATH = "dataset/onDrive-divided-cropped-augmented"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
EMBEDDING_DIM = 64
MARGIN = 1.0
BATCH_SIZE = 16
EPOCHS = 20

# ------------- FEATURE EXTRACTION -------------
def crossing_number_features(skeleton):
    rows, cols = skeleton.shape
    feature_pts = []
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0,  1), (1,  1), (1,  0), (1, -1), (0, -1)]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if skeleton[y, x] == 1:
                cn_sum = 0
                for i in range(8):
                    curr_val = skeleton[y + neighbors_offsets[i][0], x + neighbors_offsets[i][1]]
                    next_val = skeleton[y + neighbors_offsets[(i + 1) % 8][0], x + neighbors_offsets[(i + 1) % 8][1]]
                    cn_sum += abs(int(curr_val) - int(next_val))
                CN = cn_sum / 2.0
                if CN == 1 or CN == 3 or CN > 3:
                    feature_pts.append((x, y))
    return feature_pts

def angle_degrees(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    return np.degrees(np.arctan2(dy, dx))

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def build_retina_template_invariant(feature_points, k=4):
    if len(feature_points) == 0:
        return np.empty((0, 8), dtype=np.float32)
    template = np.zeros((len(feature_points), 8), dtype=np.float32)
    for i, (cx, cy) in enumerate(feature_points):
        angles = [angle_degrees(cx, cy, nx, ny)
                  for j, (nx, ny) in enumerate(feature_points) if i != j]
        base_angle = np.median(angles) if angles else 0
        dist_rel_list = []
        for j, (nx, ny) in enumerate(feature_points):
            if i == j:
                continue
            dist = euclidean_distance(cx, cy, nx, ny)
            rel_angle = (angle_degrees(cx, cy, nx, ny) - base_angle) % 360
            dist_rel_list.append((dist, rel_angle))
        dist_rel_list.sort(key=lambda x: x[0])
        nearest = dist_rel_list[:k]
        dist_vals, angle_vals = zip(*nearest) if nearest else ([], [])
        while len(dist_vals) < k:
            dist_vals += (0.0,)
            angle_vals += (0.0,)
        template[i, :k] = dist_vals
        template[i, k:] = angle_vals
    return template

def extract_template(image_path):
    vessel_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, vessel_bin = cv2.threshold(vessel_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = skeletonize((vessel_bin > 0).astype(np.uint8)).astype(np.uint8)
    feature_points = crossing_number_features(skeleton)
    return build_retina_template_invariant(feature_points, k=4)

# ------------- TRIPLET DATASET -------------
class RetinaTripletDataset(Dataset):
    def __init__(self, root):
        self.data = []
        self.labels = []
        self.label_to_paths = {}

        folders = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        for label in folders:
            paths = sorted([os.path.join(root, label, f) for f in os.listdir(os.path.join(root, label))])
            self.label_to_paths[label] = paths
            for path in paths:
                self.data.append(path)
                self.labels.append(label)

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_path = self.data[index]
        anchor_label = self.labels[index]
        anchor_template = extract_template(anchor_path)
        anchor_vec = np.mean(anchor_template, axis=0) if anchor_template.shape[0] > 0 else np.zeros(8)

        positive_path = random.choice([p for p in self.label_to_paths[anchor_label] if p != anchor_path])
        positive_template = extract_template(positive_path)
        positive_vec = np.mean(positive_template, axis=0) if positive_template.shape[0] > 0 else np.zeros(8)

        negative_label = random.choice([l for l in self.label_to_paths.keys() if l != anchor_label])
        negative_path = random.choice(self.label_to_paths[negative_label])
        negative_template = extract_template(negative_path)
        negative_vec = np.mean(negative_template, axis=0) if negative_template.shape[0] > 0 else np.zeros(8)

        return (torch.tensor(anchor_vec, dtype=torch.float32),
                torch.tensor(positive_vec, dtype=torch.float32),
                torch.tensor(negative_vec, dtype=torch.float32))

# ------------- TRIPLET NETWORK -------------
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=8, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.normalize(self.fc2(x), p=2, dim=1)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        return self.embedding_net(anchor), self.embedding_net(positive), self.embedding_net(negative)

# ------------- TRAINING LOOP -------------
def train_triplet_model():
    dataset = RetinaTripletDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TripletNet(EmbeddingNet()).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            out_a, out_p, out_n = model(anchor, positive, negative)
            loss = criterion(out_a, out_p, out_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "triplet_model.pt")
    print("Model saved as triplet_model.pt")
    return model

if __name__ == "__main__":
    train_triplet_model()
