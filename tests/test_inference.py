import pandas as pd
import pytest
from main.inference import inference

@pytest.mark.unit
def test_inference_output_type():
    result = inference(user_id=3)
    assert isinstance(result, pd.DataFrame)

@pytest.mark.unit
def test_inference_output_columns():
    result = inference(user_id=2)
    assert list(result.columns) == ['user_id', 'item_id', 'movie_title']

@pytest.mark.unit
def test_inference_number_of_recommendations():
    result = inference(user_id=20)
    assert (result['user_id'].value_counts() == 3).all()

@pytest.mark.unit
def test_inference_unknown_user():
    result = inference(user_id=99999999)
    assert result.empty or result['user_id'].isin([99999999]).all()
    