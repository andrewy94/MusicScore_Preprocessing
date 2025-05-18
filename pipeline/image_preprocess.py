import cv2
import numpy as np

def load_image(img_file):
    # Load an image file as a numpy array
    img = cv2.imread(img_file)
    if img is None:
        raise FileNotFoundError(f"Image file not found or unreadable: '{img_file}'")
    
    return img

def to_grayscale(img):
    # Convert img from colored (BGR) to grayscale (3-channel to 1-channel)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img

def binarize(gray_img):
    # Convert a 1-channel image to a binary image
    _,bin_img= cv2.threshold(gray_img,200,255,cv2.THRESH_BINARY)
    
    return bin_img

def invert(bin_img):
    inv_img = cv2.bitwise_not(bin_img)

    return bin_img

def calculate_skew_angle(bin_img):
    # Calculate skew angle using canny edge detection and standard hough line transform
    # to find lines and determine the average angle of near-horizontal lines
    blur = cv2.GaussianBlur(
        bin_img, 
        (3, 3), 
        0)
    
    edges = cv2.Canny(
        blur, 
        30, 
        100)
    
    lines = cv2.HoughLines(
        edges, 
        1, 
        np.pi / 3600, 
        200)


    angles = []
    if lines is not None:
        for _, theta in lines[:, 0]: 
            angle_deg = np.degrees(theta) - 90
            if -45 < angle_deg < 45:
                angles.append(angle_deg)

        
    if len(angles) > 0: 
        skew_angle = np.mean(angles)
        print(f"Skew angle: {skew_angle:.2f} degrees")

        if abs(skew_angle) > 1:
            pass
        
        else:
            skew_angle = 0

    else:
        skew_angle = 0
        print("No significant lines detected for deskewing.")

    return skew_angle

def deskew(bin_img, skew_angle):
    (h, w) = bin_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

    pp_img = cv2.warpAffine(bin_img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return pp_img



