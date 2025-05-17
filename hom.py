import cv2
import numpy as np

# Source points (scaled to 224x384)
src_points = np.float32([(23, 188), (371, 186), (154, 30), (232, 30)])
# Destination points (scaled to 544x960)
dst_points = np.float32([(125, 544), (835, 544), (125, 0), (835, 0)])
H = cv2.getPerspectiveTransform(src_points, dst_points)

np.save('D:/grad project/YOLOP 21/homography.npy', H)
print("Homography matrix saved as homography.npy")
print(f"Homography matrix:\n{H}")