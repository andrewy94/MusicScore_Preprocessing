import numpy as np
import cv2

def generate_dummy_score(
        width: int,
        height: int,
        num_staves: int,
        line_spacing: int,
        stave_spacing: int,
        line_thickness: int,
        left_margin: int,
        top_margin: int
):
    dummy_img = np.zeros((height, width), dtype=np.uint8)
    line_pos = []
    for stave_index in range(num_staves):
        top_y = top_margin + stave_index * stave_spacing
        for line in range(5):
            y = top_y + line * line_spacing
            line_pos.append(y)
            cv2.line(dummy_img, (left_margin, y), (width - left_margin, y), 255, thickness=line_thickness)
    return dummy_img, line_pos


