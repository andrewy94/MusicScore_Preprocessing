import cv2
import numpy as np

def connected_components_segmentation(img):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        bboxes.append((x, y, x + w, y + h))
    return bboxes

