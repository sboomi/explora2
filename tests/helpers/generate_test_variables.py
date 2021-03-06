"""
Generates test variables so the code doesn't get repeated
"""
import cv2
import re
from tangram_app.distances import detect_form
from tangram_app.processing import preprocess_img_2, crop, blur, get_contours


def generate_cnts_forms():
    img, pattern = load_samples()
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    cnts_forms = detect_form(cnts, img)
    return cnts_forms


def generate_first_ctns():
    img, pattern = load_samples()
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    img_cropped = crop(img_cv, side=side)
    image_blurred = blur(img_cropped, 1)
    first_cnts = get_contours(image_blurred)
    return first_cnts, image_blurred


def load_samples():
    img = 'data/test_images/bateau_4_right.jpg'
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    return img, pattern
