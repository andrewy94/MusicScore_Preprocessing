import cv2
import numpy as np

def detect_staves(pp_img):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(pp_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    _,detected_staves = cv2.threshold(detect_horizontal,30,255,cv2.THRESH_BINARY)

    horizontal_projection = np.sum(detected_staves, axis = 1)
    threshold = 0.9 * np.max(horizontal_projection)
    potential_lines = np.where(horizontal_projection > threshold)

    return detected_staves
    
def remove_staves(pp_img, detected_staves):
    image_without_lines = cv2.subtract(pp_img, detected_staves) 

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    repair_gaps = cv2.morphologyEx(image_without_lines, cv2.MORPH_CLOSE, vertical_kernel)    

    final_image = cv2.bitwise_not(repair_gaps)