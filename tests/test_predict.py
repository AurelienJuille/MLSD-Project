import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pandas as pd

from online_app.app import predict


def test_predict():
    fake_features = pd.Series([0.1] * 38)
    result = predict(fake_features)
    assert 0.0 <= result <= 1.0
