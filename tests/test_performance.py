import pytest
from main.inference import inference

@pytest.mark.performance
def test_inference_speed_hard_limit(benchmark):
    result = benchmark(inference, user_id=28)
    
    assert result.shape[0] <= 3
    assert list(result.columns) == ['user_id', 'item_id', 'movie_title']
    
    # Fail if it took longer than 10s
    assert benchmark.stats['mean'] < 10
