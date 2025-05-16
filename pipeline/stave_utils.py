import cv2
import numpy as np
from statistics import mode

def detect_hor_lines(pp_img):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(pp_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    _,detected_staves = cv2.threshold(detect_horizontal,30,255,cv2.THRESH_BINARY)

    horizontal_projection = np.sum(detected_staves, axis = 1)
    threshold = 0.9 * np.max(horizontal_projection)
    potential_lines = np.where(horizontal_projection > threshold)
    
    y_coords = potential_lines[0]

    return detected_staves, y_coords

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
    
def remove_hor_lines(pp_img, detected_staves):
    no_hor_img = cv2.subtract(pp_img, detected_staves) 

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    no_hor_img = cv2.morphologyEx(no_hor_img, cv2.MORPH_CLOSE, vertical_kernel)    

    return no_hor_img