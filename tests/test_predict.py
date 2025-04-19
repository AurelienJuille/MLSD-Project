import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from online_app.app import predict
import pandas as pd

def test_predict():
    fake_features = pd.Series([0.1] * 38)
    result = predict(fake_features)
    assert 0.0 <= result <= 1.0