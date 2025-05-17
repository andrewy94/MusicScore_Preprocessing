import cv2
import numpy as np

def find_segments(proj):
    coords = []
    in_segment = False
    start = 0
    for i, val in enumerate(proj):
        if val > 0 and not in_segment:
            in_segment = True
            start = i
        elif val == 0 and in_segment:
            in_segment = False
            coords.append((start, i))
    if in_segment:
        coords.append((start, len(proj)))
    return coords

def projection_segmentation(img):
    alternate_flag = 1
    prev_count = 0
    # Initial bbox: full image
    y2, x2 = img.shape
    curr_bboxes = [(0, 0, x2, y2)]

    while prev_count != len(curr_bboxes):
        prev_count = len(curr_bboxes)
        new_bboxes = []

        for x1, y1, x2, y2 in curr_bboxes:
            segment = img[y1:y2, x1:x2]

            if alternate_flag:  # horizontal
                proj = np.sum(segment, axis=1)
                segments = find_segments(proj)
                for seg in segments:
                    new_bboxes.append((x1, y1 + seg[0], x2, y1 + seg[1]))
            else:  # vertical
                proj = np.sum(segment, axis=0)
                segments = find_segments(proj)
                for seg in segments:
                    new_bboxes.append((x1 + seg[0], y1, x1 + seg[1], y2))

        curr_bboxes = new_bboxes
        alternate_flag = 1 - alternate_flag

    return curr_bboxes

def connected_component_segmentation(img):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, _ = stats[i]
        bboxes.append((x, y, x + w, y + h))
    return bboxes


