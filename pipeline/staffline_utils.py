import cv2
import numpy as np
from statistics import mode

def detect_hor_lines(pp_img):
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    detect_lines = cv2.morphologyEx(pp_img, cv2.MORPH_OPEN, hor_kernel, iterations=2)


    _,hor_lines = cv2.threshold(detect_lines,30,255,cv2.THRESH_BINARY)

    horizontal_projection = np.sum(hor_lines, axis = 1)
    threshold = 0.9 * np.max(horizontal_projection)
    potential_stafflines = np.where(horizontal_projection > threshold)
    
    y_coords = potential_stafflines[0]

    return y_coords

def average_close_hor_lines(obs_y_coords):
    y_coords = np.sort(obs_y_coords)
    lines = []
    current_line = [y_coords[0]]
    for prev, curr in zip(y_coords, y_coords[1:]):
        if curr - prev <= 3:
          current_line.append(curr)
        else:
           lines.append(current_line)
           current_line = [curr]
    
    lines.append(current_line)

    return [int(np.mean(line)) for line in lines]

def group_stave_lines(lines):
    staves = []
    i = 0
    while i + 4 < len(lines):
        window = lines[i:i+5]
        spacings = [window[j+1] - window[j] for j in range(4)]
        spacings_mode = mode(spacings)

        if all(abs(space - spacings_mode) <= 3 for space in spacings):
            staves.append(window.copy())
            i += 5
        else:
            i += 1
    
    return staves

def remove_hor_lines(pp_img, y_coords):
    h, w = pp_img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    for y in y_coords:
        rows_to_mask = [y-1, y, y+1]
        rows_to_mask = [r for r in rows_to_mask if 0 <= r < h]
        mask[rows_to_mask, :] = 255

    no_hor_img = cv2.inpaint(pp_img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return no_hor_img

def remove_vert_lines(pp_img):
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,22))
    detect_lines = cv2.morphologyEx(pp_img, cv2.MORPH_OPEN, vert_kernel, iterations=2)


    _, vert_lines = cv2.threshold(detect_lines,30,255,cv2.THRESH_BINARY)
    vert_lines = cv2.dilate(vert_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

    no_vert_img = cv2.subtract(pp_img, vert_lines)


    return no_vert_img