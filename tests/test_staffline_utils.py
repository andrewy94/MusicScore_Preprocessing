import pytest
import numpy as np
import cv2
from pipeline.staffline_utils import *
from tests.utils import generate_dummy_score

def test_detect_hor_lines_valid_img():
    # setup
    dummy_img, exp_y_coords = generate_dummy_score(
        height = 3508,
        width = 2480,
        num_staves = 8,
        line_spacing = 30,
        stave_spacing = 400,
        line_thickness = 1,
        left_margin = 100,
        top_margin = 250
    )

    # test
    act_y_coords = detect_hor_lines(dummy_img)
    obs_len = len(act_y_coords)
    exp_len = len(exp_y_coords)
    assert obs_len == exp_len, f"expected {exp_len} y_coords, but got {obs_len} y_coords"

def test_average_close_hor_lines_valid_img():
    # setup
    dummy_img, exp_y_coords = generate_dummy_score(
        height = 3508,
        width = 2480,
        num_staves = 8,
        line_spacing = 30,
        stave_spacing = 400,
        line_thickness = 2,
        left_margin = 100,
        top_margin = 250
    )
    act_y_coords = detect_hor_lines(dummy_img)

    # test
    merged_obs_y_coords = average_close_hor_lines(act_y_coords)

    obs_len = len(merged_obs_y_coords)
    exp_len = len(exp_y_coords)

    assert obs_len == exp_len, f"expected {exp_len} y_coords, but got {obs_len} y_coords"

def test_group_stave_lines_valid_img():
    # setup
    num_staves = 8
    dummy_img, _ = generate_dummy_score(
        height = 3508,
        width = 2480,
        num_staves = num_staves,
        line_spacing = 30,
        stave_spacing = 400,
        line_thickness = 2,
        left_margin = 100,
        top_margin = 250
    )
    act_y_coords = detect_hor_lines(dummy_img)
    merged_obs_y_coords = average_close_hor_lines(act_y_coords)

    # test
    staves = group_stave_lines(merged_obs_y_coords)
    staves_len = len(staves)

    assert num_staves == staves_len, f"expected {num_staves} staves, but got {staves_len} staves"

def test_remove_hor_lines():
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
    y_coords = detect_hor_lines(dummy_img)

    # test
    dummy_img_rmv_lines = remove_hor_lines(dummy_img, y_coords)

    obs_y_coords_post_rmvl = detect_hor_lines(dummy_img_rmv_lines)

    assert len(obs_y_coords_post_rmvl) == 0, f"expected 0 lines, but got {len(obs_y_coords_post_rmvl)} lines"

def test_remove_vert_lines():
    # setup
    dummy_img = np.zeros((1000, 1000), dtype = np.uint8)

    lines = [
        ((100, 50), (100, 75)),
        ((200, 50), (200, 125)),
        ((300, 50), (300, 200)),
        ((400, 50), (400, 400)),
        ((500, 50), (500, 500)),
        ((600, 50), (600, 600)),
        ((700, 50), (700, 700)),
        ((800, 50), (800, 800)),
        ((900, 50), (900, 950))
    ]
    for start, end in lines:
        cv2.line(dummy_img, start, end, 255, thickness = 1)

    cv2.imshow("dummy", dummy_img)
    cv2.waitKey(0)

    # test
    no_vert_img = remove_vert_lines(dummy_img)

    cv2.imshow("no vert", no_vert_img)
    cv2.waitKey(0)

    assert np.count_nonzero(no_vert_img) == 26, f"expected only {lines[0]} line to survive, but other lines survived"

    