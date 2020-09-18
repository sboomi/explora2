"""
@author: Renata
"""

import re
import cv2
import numpy as np
from tangram_app.processing import *
from tests.helpers.generate_test_variables import generate_first_ctns


def test_preprocess_img():
    img = 'data/test_images/bateau_4_right.jpg'

    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img(img_cv, side=side)
    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)


def test_preprocess_img_2():
    img = 'data/test_images/bateau_4_right.jpg'

    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)


def test_extract_triangles_squares():
    first_cnts, image_blurred = generate_first_ctns()
    img = extract_triangles_squares(first_cnts, image_blurred)
    assert isinstance(img, np.ndarray)


def test_blur():
    img = 'data/test_images/bateau_4_right.jpg'
    img_cv = cv2.imread(img)
    image_blurred = blur(img_cv, 3)
    assert isinstance(image_blurred, np.ndarray)


def test_get_contours():
    cnts = generate_first_ctns()
    assert isinstance(cnts, list)


def test_extract_triangles_squares_2():
    first_cnts, image_blurred = generate_first_ctns()
    cnts, img = extract_triangles_squares_2(first_cnts, image_blurred)

    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)


def test_crop():
    img = 'data/test_images/bateau_4_right.jpg'

    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    previous_size = img_cv.shape
    print(previous_size)

    img = crop(img_cv, side=side)

    assert previous_size > img.shape, "The cropping didn't occur"
