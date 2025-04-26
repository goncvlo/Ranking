import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_joblib_load():
    with patch("main.inference.joblib.load") as mock_load:
        # Setup the mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5] * 3  # or whatever fake scores you like
        mock_load.return_value = mock_model
        yield mock_load
