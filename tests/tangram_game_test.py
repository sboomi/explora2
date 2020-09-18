"""
@author: Renata
"""

import pytest
import pandas as pd
from tangram_app.tangram_game import preprocess_img_2,tangram_game
from tangram_app.predictions import get_predictions_with_distances


def test_tangram_game():
    # test the probabilities of the image / frame
    path = "data/test_images/bateau_4_right.jpg"
    probability = tangram_game(image=path, prepro=preprocess_img_2, pred_func=get_predictions_with_distances)
    assert isinstance(probability, pd.core.frame.DataFrame), 'Predictions should be dataframe'
    assert probability.loc[0, 'target'] == 'bateau', 'Predictions should be bateau'
