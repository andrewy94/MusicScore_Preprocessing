import pytest
import cv2
import numpy as np
from tests.utils import generate_dummy_score, verify_bboxes
from pipeline.segmentation import find_segments, projection_segmentation, connected_component_segmentation

# baseline test
def test_find_segments_one_pixel():
    # setup
    dummy_img = np.zeros((100, 100), dtype = np.uint8)
    dummy_img[50, 50] = 255

    # test
    hor_proj = np.sum(dummy_img, axis = 1)
    hor_segment_coords = find_segments(hor_proj)
    vert_proj = np.sum(dummy_img, axis = 1)
    vert_segment_coords = find_segments(vert_proj)

    assert len(hor_segment_coords) == 1, f"Expected 1 segment, but got {len(hor_segment_coords)} segments "
    assert len(vert_segment_coords) == 1, f"Expected 1 segment, but got {len(vert_segment_coords)} segments "

# test one-pixel spacing behaviour
def test_find_segments_two_pixels_one_pixel_apart():
    dummy_img = np.zeros((100, 100), dtype = np.uint8)
    dummy_img[50, 49] = 255
    dummy_img[50, 51] = 255

    # test
    hor_proj = np.sum(dummy_img, axis = 1)
    hor_segment_coords = find_segments(hor_proj)
    vert_proj = np.sum(dummy_img, axis = 0)
    vert_segment_coords = find_segments(vert_proj)

    assert len(hor_segment_coords) == 1, f"Expected 40 segments, but got {len(hor_segment_coords)} segments " # 5 lines in a stave * 8
    assert len(vert_segment_coords) == 2, f"Expected 1 segment, but got {len(vert_segment_coords)} segments "

# simulated score test
def test_find_segments_eight_staves():
    # setup
    dummy_img, _ = generate_dummy_score(
        height = 3508,
        width = 2480,
        num_staves = 8,
        line_spacing = 30,
        stave_spacing = 400,
        line_thickness = 2,
        left_margin = 100,
        top_margin = 250
    )

    # test
    hor_proj = np.sum(dummy_img, axis = 1)
    hor_segment_coords = find_segments(hor_proj)
    vert_proj = np.sum(dummy_img, axis = 0)
    vert_segment_coords = find_segments(vert_proj)

    assert len(hor_segment_coords) == 40, f"Expected 40 segments, but got {len(hor_segment_coords)} segments " # 5 lines in a stave * 8
    assert len(vert_segment_coords) == 1, f"Expected 1 segment, but got {len(vert_segment_coords)} segments "

# simple image
def test_projection_segmentation_low_depth():
    # setup
    dummy_img = np.zeros((1000, 1000), dtype = np.uint8)
    dummy_img[500, 500] = 255 # middle dot
    lines = [
        ((200, 250), (200, 750)),
        ((250, 200), (750, 200)),
        ((800, 250), (800, 750)),
        ((750, 800), (250, 800))
    ]
    for start, end in lines:
        cv2.line(dummy_img, start, end, 255, thickness = 1)

    # test
    exp_bboxes = [(250, 200, 751, 201), 
                  (200, 250, 201, 751), 
                  (500, 500, 501, 501), 
                  (800, 250, 801, 751), 
                  (250, 800, 751, 801)]
    
    act_bboxes = projection_segmentation(dummy_img)

    assert act_bboxes == exp_bboxes, f"Expected bboxes {exp_bboxes}, but got {act_bboxes}"

# complex image   
def test_projection_segmentation_moderate_depth():
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
                  (800, 200, 801, 201), (200, 250, 201, 751), (300, 300, 401, 401), 
                  (300, 600, 401, 701), (500, 500, 501, 501), (505, 500, 506, 501), 
                  (540, 650, 541, 651), (600, 300, 701, 401), (600, 600, 701, 726), 
                  (800, 250, 801, 751), (200, 800, 201, 801), (250, 800, 751, 801), 
                  (800, 800, 801, 801), (100, 900, 101, 901)]
    
    act_bboxes = projection_segmentation(dummy_img)

    # cv2.imshow("img", verify_bboxes(dummy_img, act_bboxes))
    # cv2.waitKey(0)
    assert act_bboxes == exp_bboxes, f"Expected bboxes {exp_bboxes}, but got {act_bboxes}"

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
    
    act_bboxes = connected_component_segmentation(dummy_img)

    assert act_bboxes == exp_bboxes, f"Expected bboxes {exp_bboxes}, but got {act_bboxes}"
