import pytest
from main.inference import inference

@pytest.mark.performance
def test_inference_speed_hard_limit(benchmark):
    result = benchmark(inference, user_ids=[1, 2, 3])
    
    assert result.shape[0] <= 9
    assert list(result.columns) == ['user_id', 'item_id', 'score']
    
    # Fail if it took longer than 10s
    assert benchmark.stats['mean'] < 10
