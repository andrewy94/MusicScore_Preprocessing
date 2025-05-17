import pytest
import cv2
import numpy as np
from pipeline.segmentation import connected_components_segmentation

def test_connected_component_segmentation():
    # setup
    dummy_img = np.zeros((1000, 1000), dtype = np.uint8)
    dots = [
        (500, 500),
        (200, 200),
        (800, 200),
        (200, 800),
        (800, 800),
        (650, 540),
        (100, 100),
        (900, 100),
        (650, 675),
        (500, 505)
    ]
    lines = [
        ((200, 250), (200, 750)),
        ((250, 200), (750, 200)),
        ((800, 250), (800, 750)),
        ((750, 800), (250, 800)),
        ((300, 300), (400, 400)),
        ((700, 300), (600, 400)),
        ((300, 700), (400, 600)),
        ((700, 700), (600, 600)),
        ((650, 675), (650, 725))
    ]

    for x, y in dots:
        dummy_img[x, y] = 255

    for start, end in lines:
        cv2.line(dummy_img, start, end, 255, thickness = 1)

    # test
    exp_bboxes = [(100, 100, 101, 101), (200, 200, 201, 201), (250, 200, 751, 201), 
                  (800, 200, 801, 201), (200, 250, 201, 751), (800, 250, 801, 751), 
                  (300, 300, 401, 401), (600, 300, 701, 401), (500, 500, 501, 501), 
                  (505, 500, 506, 501), (300, 600, 401, 701), (600, 600, 701, 701), 
                  (540, 650, 541, 651), (675, 650, 676, 651), (650, 675, 651, 726), 
                  (200, 800, 201, 801), (250, 800, 751, 801), (800, 800, 801, 801), 
                  (100, 900, 101, 901)]
    
    act_bboxes = connected_components_segmentation(dummy_img)

    assert act_bboxes == exp_bboxes, f"Expected bboxes {exp_bboxes}, but got {act_bboxes}"
