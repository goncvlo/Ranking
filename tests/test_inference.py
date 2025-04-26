import pandas as pd
import pytest
from main.inference import inference

def test_inference_output_type():
    result = inference(user_ids=[1, 2, 3])
    assert isinstance(result, pd.DataFrame)

def test_inference_output_columns():
    result = inference(user_ids=[1, 2])
    assert list(result.columns) == ['user_id', 'item_id', 'score']

def test_inference_number_of_recommendations():
    result = inference(user_ids=[10, 20])
    assert (result['user_id'].value_counts() == 3).all()

def test_inference_empty_input():
    result = inference(user_ids=[])
    assert result.empty

def test_inference_unknown_user():
    result = inference(user_ids=[99999999])
    assert result.empty or result['user_id'].isin([99999999]).all()
