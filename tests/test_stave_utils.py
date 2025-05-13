import pytest
import numpy as np
import cv2
from pipeline.stave_utils import detect_hor_lines, average_close_hor_lines, group_stave_lines, remove_hor_lines
from tests.utils import generate_dummy_score

def test_detect_hor_lines_valid_img():
    # setup
    dummy_img, act_y_coords = generate_dummy_score(
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
    _, obs_y_coords = detect_hor_lines(dummy_img)
    obs_len = len(obs_y_coords)
    act_len = len(act_y_coords)
    assert obs_len == act_len, f"expected {act_len} y_coords, but got {obs_len} y_coords"

def test_average_close_hor_lines_valid_img():
    # setup
    dummy_img, act_y_coords = generate_dummy_score(
        height = 3508,
        width = 2480,
        num_staves = 8,
        line_spacing = 30,
        stave_spacing = 400,
        line_thickness = 2,
        left_margin = 100,
        top_margin = 250
    )
    _, obs_y_coords = detect_hor_lines(dummy_img)

    # test
    merged_obs_y_coords = average_close_hor_lines(obs_y_coords)

    obs_len = len(merged_obs_y_coords)
    act_len = len(act_y_coords)

    assert obs_len == act_len, f"expected {act_len} y_coords, but got {obs_len} y_coords"

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
    _, obs_y_coords = detect_hor_lines(dummy_img)
    merged_obs_y_coords = average_close_hor_lines(obs_y_coords)

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
    detected_lines, _ = detect_hor_lines(dummy_img)

    # test
    dummy_img_rmv_lines = remove_hor_lines(dummy_img, detected_lines)

    _, obs_y_coords_post_rmvl = detect_hor_lines(dummy_img_rmv_lines)

    assert len(obs_y_coords_post_rmvl) == 0, f"expected 0 lines, but got {len(obs_y_coords_post_rmvl)} lines"

