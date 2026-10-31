"""Reload a trusted test panel and bundle in a fresh CPU process."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demand_forecast.inference.predictor import Predictor


def main():
    torch.set_num_threads(1)
    bundle, panel_path, origin, output = sys.argv[1:]
    with Path(panel_path).open("rb") as stream:
        panel = pickle.load(stream)
    _, predictions = Predictor.from_bundle(bundle).forecast_arrays(panel, pd.Timestamp(origin))
    np.save(output, predictions)


if __name__ == "__main__":
    main()
