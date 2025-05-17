import cv2
import numpy as np
import torch
import torchvision
from scipy.ndimage import label

class Lane:
    def __init__(self, lane_line):
        self.is_right = False
        self.lane_mask = self.interpolate(lane_line)
        print(f"Initial lane mask shape: {self.lane_mask.shape}, Min: {self.lane_mask.min()}, Max: {self.lane_mask.max()}")
        self.detected_lanes = self.detect_lanes()
        self.roads = np.zeros_like(self.detected_lanes, dtype=np.uint8)

    def interpolate(self, lane_line):
        ll_predict = lane_line[:, :, 12:372, :]  # Crop width
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear', align_corners=False)
        lane_mask = torch.round(ll_seg_mask).squeeze(1).int().squeeze().cpu().numpy()
        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        print(f"Interpolated lane mask shape: {lane_mask.shape}, Unique values: {np.unique(lane_mask)}")
        return lane_mask

    def detect_lanes(self):
        h = self.lane_mask.shape[0]
        self.lane_mask[:int(h * 0.3), :] = 0  # Reduced to 30% to preserve more lanes
        print(f"Lane mask after clearing top: Unique values: {np.unique(self.lane_mask)}")

        self.labeled_lanes, num_lanes = label(self.lane_mask)
        print(f"Labeled lanes: {num_lanes} components")

        H, W = self.labeled_lanes.shape
        center_col = W // 2
        start_row = int(h * 0.3)

        self.left_lanes = self.find_line_labels(start_row, center_col - 1, -1, -1)
        self.right_lanes = self.find_line_labels(start_row, center_col + 1, W, 1)

        if len(self.right_lanes) == 1:
            self.is_right = True

        self.lanes_idx = list(dict.fromkeys(self.left_lanes + self.right_lanes))
        self.lanes_idx.sort()

        self.detected_lanes = np.zeros_like(self.labeled_lanes, dtype=np.uint8)
        for idx, label_id in enumerate(self.lanes_idx[:4]):
            self.detected_lanes[self.labeled_lanes == label_id] = idx + 1

        print(f"Total labeled lanes: {num_lanes}")
        print(f"Left labels: {self.left_lanes}, Right labels: {self.right_lanes}")
        print(f"Detected {len(self.lanes_idx)} unique lanes: {self.lanes_idx[:4]}")
        return self.detected_lanes

    def find_line_labels(self, start_row, start, end, step):
        found = set()
        for col in range(start, end, step):
            labels = set(self.labeled_lanes[start_row:, col]) - {0}
            for lbl in labels:
                if lbl not in found:
                    found.add(lbl)
                    if len(found) >= 4:  # Increased to 4 labels per side
                        return list(found)
        return list(found)

    def show_detected_lanes(self, img):
        lane_colors = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
        ])
        lane_mask = np.zeros_like(img, dtype=np.uint8)

        if self.detected_lanes.shape[:2] != img.shape[:2]:
            self.detected_lanes = cv2.resize(
                self.detected_lanes,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        unique_labels = np.unique(self.detected_lanes)
        unique_labels = unique_labels[unique_labels != 0][:4]

        label_to_color = {label: lane_colors[idx % len(lane_colors)] for idx, label in enumerate(unique_labels)}

        for label, color in label_to_color.items():
            lane_mask[self.detected_lanes == label] = color

        output_img = img.astype(np.float32)
        output_img[lane_mask != 0] = output_img[lane_mask != 0] * 0.5 + lane_mask[lane_mask != 0] * 0.5
        return output_img.astype(np.uint8)

    def detect_road(self):
        num_lanes = len(self.lanes_idx)
        if num_lanes < 2:
            print(f"[⚠️ Lane Warning] Only {num_lanes} unique lanes detected, skipping road region filling.")
            return

        # Handle 2 or more lanes dynamically
        for i in range(num_lanes - 1):
            lane1_dict = self.rowwise_col_stat_for_loop(self.detected_lanes, i + 1, 'max')
            lane2_dict = self.rowwise_col_stat_for_loop(self.detected_lanes, i + 2, 'min')
            self.fill_between(lane1_dict, lane2_dict, i + 1)

    def fill_between(self, dict1, dict2, idx):
        for row in dict1:
            if row in dict2:
                col1 = dict1[row]
                col2 = dict2[row]
                cmin, cmax = min(col1, col2), max(col1, col2)
                self.roads[row, cmin:cmax] = idx

    def rowwise_col_stat_for_loop(self, label_mask, target_label, stat='min'):
        coords = np.where(label_mask == target_label)
        rows = coords[0]
        cols = coords[1]
        row_dict = {}
        for r, c in zip(rows, cols):
            if r not in row_dict:
                row_dict[r] = c
            else:
                if stat == 'min':
                    row_dict[r] = min(row_dict[r], c)
                elif stat == 'max':
                    row_dict[r] = max(row_dict[r], c)
        return row_dict

    def show_roads(self, img):
        img_float = img.astype(np.float32)
        if self.roads.shape[:2] != img.shape[:2]:
            self.roads = cv2.resize(
                self.roads,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        colors = {
            1: np.array([255, 0, 0], dtype=np.uint8),   # Red
            2: np.array([0, 255, 0], dtype=np.uint8),   # Green
            3: np.array([255, 0, 0], dtype=np.uint8),   # Red again
        }
        overlay = np.zeros_like(img, dtype=np.uint8)
        for value, color in colors.items():
            overlay[self.roads == value] = color
        blended = (img_float + overlay.astype(np.float32) * 0.5).astype(np.uint8)
        return blended