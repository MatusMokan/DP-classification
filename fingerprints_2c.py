import numpy as np
import cv2
from skimage.morphology import skeletonize

def extract_retina_features(vessel_gray, point_size=3):
    """
    1) Skeletonize the binarized vessel image
    2) Compute crossing number for each vessel pixel
    3) Mark endpoints, bifurcations, and crossings with large colored circles
    4) Return (skeleton, color-coded features) for display
    -------------------------------------------------------
    vessel_img: 2D NumPy array (0 or 255) or (0 or 1)
    point_size: Radius of circles to draw for feature points
    """

    # --- (0) Otsu Threshold ---
    # THRESH_OTSU automatically finds an optimal threshold value.
    # We'll use threshold=0 (ignored), maxValue=255
    _, vessel_img = cv2.threshold(vessel_gray, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 1) Ensure vessel_img is binary in {0,1}, then skeletonize
    bin_img = (vessel_img > 0).astype(np.uint8)
    skeleton = skeletonize(bin_img).astype(np.uint8)

    # Prepare 3-channel color image for superimposing features
    # Start by stacking the skeleton in all 3 channels (white skeleton)
    skeleton_3ch = np.dstack([skeleton*255, skeleton*255, skeleton*255])

    # Make feature map a bit larger for better visibility (black background)
    color_features = np.zeros_like(skeleton_3ch)
    # Add skeleton in white
    color_features[skeleton == 1] = [255, 255, 255]

    # Store feature points to draw circles later
    endpoints = []
    bifurcations = []
    crossings = []

    # Offsets of 8-neighbors in clockwise order:
    # (p1) top-left, (p2) top, (p3) top-right, (p4) right,
    # (p5) bottom-right, (p6) bottom, (p7) bottom-left, (p8) left
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0,  1),
                         (1,  1), (1,  0), (1, -1),
                         (0, -1)]

    # 3) For each pixel in skeleton, compute Crossing Number (CN)
    rows, cols = skeleton.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            if skeleton[y, x] == 1:
                # Gather 8 neighbors in a clockwise loop
                cn_sum = 0
                for i in range(8):
                    current_val = skeleton[y + neighbors_offsets[i][0],
                                           x + neighbors_offsets[i][1]]
                    next_val = skeleton[y + neighbors_offsets[(i+1)%8][0],
                                        x + neighbors_offsets[(i+1)%8][1]]
                    cn_sum += abs(int(current_val) - int(next_val))

                CN = cn_sum / 2.0

                # 4) Classify feature points using CN:
                #    - End point (CN == 1) -> RED
                #    - Bifurcation (CN == 3) -> GREEN
                #    - Crossing (CN > 3) -> BLUE
                if CN == 1:
                    endpoints.append((x, y))
                elif CN == 3:
                    bifurcations.append((x, y))
                elif CN > 3:
                    crossings.append((x, y))
    
    # Draw circles for each feature point
    for x, y in endpoints:
        cv2.circle(color_features, (x, y), point_size, (0, 0, 255), -1)  # Red filled circle
    
    for x, y in bifurcations:
        cv2.circle(color_features, (x, y), point_size, (0, 255, 0), -1)  # Green filled circle
    
    for x, y in crossings:
        cv2.circle(color_features, (x, y), point_size, (255, 0, 0), -1)  # Blue filled circle

    return vessel_img, skeleton_3ch, color_features
    
# ----------------------- Example Usage -----------------------
if __name__ == "__main__":
    # 1) Load a binary vessel image (0 or 255). 
    #    Suppose "vessel.png" is a pre-extracted vessel map.
    vessel_img = cv2.imread("dataset/FIRE/onDrive-divided-augmented-res/person_A01/A01_1_seg_dark_neg_rot.png", cv2.IMREAD_GRAYSCALE)
    # vessel_img = cv2.imread("dataset/onDrive/IM000001_1_bin_seg.png", cv2.IMREAD_GRAYSCALE)

    # Make sure vessel_img is thresholded/binary, e.g., using Otsu if needed:
    # _, vessel_bin = cv2.threshold(vessel_img, 128, 255, cv2.THRESH_OTSU)

    # 2) Extract skeleton and color-coded feature map
    vessel_img, skeleton_rgb, feature_map = extract_retina_features(vessel_img, point_size=5)  # Adjust point size here

    # 3) Display or save results
    cv2.imshow("1) Binary Segmented Image (Otsu)", vessel_img)

    cv2.imshow("Skeleton (White)", skeleton_rgb)
    cv2.imshow("Feature Map (Endpoints/Bifurcations/Crossings)", feature_map)
    
    # Save the visualization
    cv2.imwrite("skeleton.png", skeleton_rgb)
    cv2.imwrite("feature_map.png", feature_map)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from skimage.morphology import skeletonize

# import numpy as np
# import math
# from sklearn.decomposition import PCA


# def euclidean_distance(x1, y1, x2, y2):
#     """Compute Euclidean distance between two points (x1,y1) and (x2,y2)."""
#     return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# def angle_degrees(x1, y1, x2, y2):
#     """
#     Returns the angle (in degrees) from (x1,y1) to (x2,y2) in the range [-180, 180].
#     Uses math.atan2(dy, dx).
#     """
#     dx = x2 - x1
#     dy = y2 - y1
#     angle_rad = math.atan2(dy, dx)  # range: [-pi, pi]
#     angle_deg = math.degrees(angle_rad)  # range: [-180, 180]
#     return angle_deg

# def crossing_number_features(skeleton):
#     """
#     Return a list of (x, y) feature points from a skeletonized image,
#     using crossing-number technique to detect endpoints, bifurcations, crossing-over.
#     """
#     rows, cols = skeleton.shape
#     feature_pts = []
#     # 8-neighbors in clockwise order
#     neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
#                          (0,  1),
#                          (1,  1), (1,  0), (1, -1),
#                          (0, -1)]
#     for y in range(1, rows-1):
#         for x in range(1, cols-1):
#             if skeleton[y, x] == 1:
#                 # compute crossing number
#                 cn_sum = 0
#                 for i in range(8):
#                     curr_val = skeleton[y + neighbors_offsets[i][0],
#                                        x + neighbors_offsets[i][1]]
#                     next_val = skeleton[y + neighbors_offsets[(i+1)%8][0],
#                                        x + neighbors_offsets[(i+1)%8][1]]
#                     cn_sum += abs(int(curr_val) - int(next_val))

#                 CN = cn_sum / 2.0

#                 # If you want *all* endpoints/bifurcations/crossings:
#                 if CN == 1 or CN == 3 or CN > 3:
#                     feature_pts.append((x, y))
#     return feature_pts

# def build_retina_template(feature_points, k=4):
#     """(Same function as above)"""
#     if len(feature_points) == 0:
#         return np.empty((0, 8), dtype=np.float32)

#     template = np.zeros((len(feature_points), 8), dtype=np.float32)
#     for i, (cx, cy) in enumerate(feature_points):
#         # distances to all others
#         dist_list = []
#         for j, (nx, ny) in enumerate(feature_points):
#             if i == j: continue
#             dist = euclidean_distance(cx, cy, nx, ny)
#             dist_list.append((dist, nx, ny))

#         dist_list.sort(key=lambda x: x[0])
#         nearest = dist_list[:k]

#         dist_vals, angle_vals = [], []
#         for (dist, nx, ny) in nearest:
#             dist_vals.append(dist)
#             angle_vals.append(angle_degrees(cx, cy, nx, ny))

#         # Zero-padding if < k neighbors
#         while len(dist_vals) < k:
#             dist_vals.append(0.0)
#             angle_vals.append(0.0)

#         template[i, :k] = dist_vals
#         template[i, k:] = angle_vals
#     return template

# if __name__ == "__main__":
#     # 1) Load a grayscale retina vessel image
#     vessel_gray = cv2.imread("dataset/FIRE/onDrive-divided-augmented-res/person_A01/A01_1_seg_dark_neg_rot.png", cv2.IMREAD_GRAYSCALE)

#     # vessel_gray = cv2.imread("retina_vessel.png", cv2.IMREAD_GRAYSCALE)
#     if vessel_gray is None:
#         raise FileNotFoundError("Could not load retina_vessel.png")

#     # 2) Binarize (Otsu threshold)
#     _, vessel_bin = cv2.threshold(vessel_gray, 0, 255, 
#                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 3) Skeletonize
#     bin_img = (vessel_bin > 0).astype(np.uint8)
#     skeleton = skeletonize(bin_img).astype(np.uint8)

#     # Display the skeletonized image
#     cv2.imshow("Skeletonized Image", skeleton * 255)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # 4) Extract feature points (endpoints, bifurcations, crossing-over)
#     feature_points = crossing_number_features(skeleton)

#     # 5) Build Nx8 template
#     retina_template = build_retina_template(feature_points, k=4)

#     print(f"Extracted {len(feature_points)} feature points.")
#     print(f"Template shape: {retina_template.shape}")  # (R, 8)

#     # 6) (Optional) PCA or store in "database"
#     # Assuming retina_template is already computed with shape (R, 8)
#     print("Original template shape:", retina_template.shape)

#     # Apply PCA to reduce from 8 dimensions to 4 dimensions
#     pca = PCA(n_components=4)
#     reduced_template = pca.fit_transform(retina_template)

#     print("Reduced template shape:", reduced_template.shape)
#     print("Explained variance ratio:", pca.explained_variance_ratio_)