"""
@author: Renata
"""

from tangram_app.utils import *


def test_get_files():
    assert len(get_files(directory='data/tangrams')) == 12, 'Images must be 12'
