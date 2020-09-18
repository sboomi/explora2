"""
@author: Renata
"""

import pytest
from tangram_app.utils import get_files


def test_get_files():
    assert len(get_files(directory='data/tangrams')) == 12, 'Images must be 12'
