"""
@author: Renata
"""

import pytest
import re
import cv2
import numpy as np
from tangram_app.processing import preprocess_img, preprocess_img_2, extract_triangles_squares, blur, \
    extract_triangles_squares_2, crop
from tests.helpers.generate_test_variables import generate_first_ctns, load_samples


def test_preprocess_img():
    img, pattern = load_samples()
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img(img_cv, side=side)
    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)


def test_preprocess_img_2():
    img, pattern = load_samples()
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
    img, _ = load_samples()
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
    img, pattern = load_samples()
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    previous_size = img_cv.shape
    print(previous_size)

    img = crop(img_cv, side=side)

    assert previous_size > img.shape, "The cropping didn't occur"
