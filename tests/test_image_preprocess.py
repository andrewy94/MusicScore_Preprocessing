import pytest
import numpy as np
import cv2
from pipeline.image_preprocess import *

def test_load_image_valid_path(tmp_path):
    dummy_img = np.zeros((100, 100, 3))
    dummy_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(dummy_path), dummy_img)
    img = load_image(str(dummy_path))

    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100, 3)

def test_load_image_no_image():
    with pytest.raises(FileNotFoundError):
        load_image("path/nonexistent.jpg")

def test_to_grayscale():
    dummy_img = np.zeros((100, 100, 3), dtype = np.uint8)
    gray_img = to_grayscale(dummy_img)

    assert gray_img.ndim == 2
    assert gray_img.shape == (100, 100)
    assert gray_img.dtype == np.uint8

def test_binarize():
    dummy_img = np.full((100, 100), 127, dtype = np.uint8)
    bin_img = binarize(dummy_img)

    assert bin_img.ndim == 2
    assert bin_img.shape == (100, 100)
    assert bin_img.dtype == np.uint8
    assert set(np.unique(bin_img)).issubset({0, 255})

def test_calculate_skew_angle_valid_lines():
    dummy_img = np.zeros((100, 300), dtype = np.uint8)
    cv2.line(dummy_img, (0, 45), (300, 50), 255, 1) # 42.3 degree angle line
    cv2.line(dummy_img, (0, 50), (300, 60), 255, 1) # 42.3 degree angle line
    cv2.line(dummy_img, (0,55), (300, 70), 255, 1) # 42.3 degree angle line
   
    skew_angle = calculate_skew_angle(dummy_img)

    assert np.isclose(skew_angle, 2, atol=2), \
        f"Expected skew angle close to 2, but got {skew_angle} degrees."

def test_calculate_skew_angle_45_deg_lines():
    dummy_img = np.zeros((300, 300), dtype = np.uint8)
    cv2.line(dummy_img, (0, 0), (300, 300), 255, 1) # 45 degree angle line
    cv2.line(dummy_img, (0, 300), (300, 0), 255, 1) # -45 degree angle line
    skew_angle = calculate_skew_angle(dummy_img)

    assert np.isclose(skew_angle, 45, atol=2), \
        f"Expected skew angle close to 45, but got {skew_angle} degrees."

def test_deskew():
    skew_angle = 45
    dummy_img = np.zeros((100, 100),dtype = np.uint8)
    cv2.line(dummy_img, (0,0), (100, 100), 255, 2) # 45 degree angle line
    deskew_img = deskew(dummy_img, skew_angle)
    post_deskew_angle = calculate_skew_angle(deskew_img)
    
    assert np.isclose(post_deskew_angle, 0, atol=2), \
        f"Expected skew angle close to 0, but got {post_deskew_angle} degrees."

