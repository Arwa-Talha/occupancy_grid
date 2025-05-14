import numpy as np
bev_map = np.load(r'D:\grad project\final_map\output\bev_map_605.npy')
print(bev_map.shape)  # Should be (500, 400)
print(np.unique(bev_map))  # Should show [0, 1, 2, 3]
def visualize_bev_map(bev_map):
    color_map = np.zeros((bev_map.shape[0], bev_map.shape[1], 3), dtype=np.uint8)
    color_map[bev_map == 1] = [255, 255, 0]  # Yellow for lanes
    color_map[bev_map == 2] = [0, 255, 0]    # Green for drivable
    color_map[bev_map == 3] = [255, 0, 0]    # Red for obstacles
    return color_map

import cv2
color_map = visualize_bev_map(bev_map)
cv2.imwrite('bev_visualization.png', color_map)
cv2.imshow("BEV Map", color_map)
cv2.waitKey(0)
cv2.destroyAllWindows()