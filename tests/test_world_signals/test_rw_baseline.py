import numpy as np

from scripts.world_signals.rw_baseline import rw_window_predictions


def test_rw_predictions_shape_and_values():
    predictions = rw_window_predictions(["BTC", "GOLD"], 13, [0.1, 0.5, 0.9])
    assert len(predictions) == 2
    for row in predictions:
        assert set(row) == {"predictions", "0.1", "0.5", "0.9"}
        for values in row.values():
            np.testing.assert_array_equal(values, np.zeros(13))
