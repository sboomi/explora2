"""
Created on Tue Jul 28 15:11:06 2020

@author: Renata
"""

from tangram_app.distances import *
from .helpers.generate_test_variables import generate_cnts_forms
import os
import pandas as pd

def test_dist_humoment():
    assert dist_humoment(0.026322267294741648,
                         0.0027161374010481088) == 0.02360612989369354, 'distance_humoment not working'


def test_detect_forme():
    cnts_forms = generate_cnts_forms()
    assert isinstance(cnts_forms, list), "the cnts_forms format isn't correct"


def test_distance_formes():
    cnts_forms = generate_cnts_forms()
    centers, perimeters = distance_formes(cnts_forms)
    print(centers)
    assert isinstance(centers, dict), "centers should be a dict"
    assert isinstance(perimeters, dict), "centers should be a dict"


def test_ratio_distance():
    cnts_forms = generate_cnts_forms()
    centers, perimeters = distance_formes(cnts_forms)
    distances = ratio_distance(centers, perimeters)
    assert isinstance(distances, dict), "the distances should be stored inside a dict"


def test_sorted_distances():
    cnts_forms = generate_cnts_forms()
    centers, perimeters = distance_formes(cnts_forms)
    distances = ratio_distance(centers, perimeters)
    sorted_dists = sorted_distances(distances)
    assert isinstance(sorted_dists, dict), "the sorted distances should be stored inside a dict"


def test_create_all_types_distances():
    create_all_types_distances("tests/data/data.csv")
    assert os.path.exists('tests/data/data.csv'), "the data csv with the distances should exist"


def test_mse_distances():
    data = pd.read_csv("data/tangram_properties/data.csv", sep=";")

    cnts_forms = generate_cnts_forms()
    centers, perimeters = distance_formes(cnts_forms)
    distances = ratio_distance(centers, perimeters)
    sorted_dists = sorted_distances(distances)

    # get mses
    mses = mse_distances(data, sorted_dists)
    assert isinstance(mses, list), "The mses should be stored in list"
    assert len(mses) == 12, "the MSES list should have a length of 12"
