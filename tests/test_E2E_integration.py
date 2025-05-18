import pytest
import cv2

from pipeline.image_preprocess import *
from pipeline.staffline_utils import detect_hor_lines, remove_hor_lines, remove_vert_lines
from pipeline.segmentation import *
from tests.utils import verify_bboxes

def test_end_to_end_segmentation_mitski_MLMAM_1():
    # image preprocessing
    img = load_image("tests/test_data/mitski_MLMAM_2.png")
    cv2.imshow("image", img)
    cv2.waitKey(0)

    gray_img = to_grayscale(img)
    bin_img = binarize(gray_img)
    inv_img = invert(bin_img)

    # stave removal
    y_coords = detect_hor_lines(inv_img)
    no_hor_img = remove_hor_lines(inv_img, y_coords)
    no_vert_img = remove_vert_lines(no_hor_img)
    processed_img = binarize(no_vert_img)

    # segmentation
    cc_segmented_img = verify_bboxes(processed_img, connected_components_segmentation(processed_img))
    cv2.imshow("cc", cc_segmented_img)
    cv2.waitKey(0)




 


 



