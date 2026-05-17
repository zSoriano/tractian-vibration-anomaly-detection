from datetime import timedelta
from pathlib import Path
from typing import List

import yaml

from .alert_engine import AlertDecision, AlertEngine
from .anomaly_model import AnomalyModel
from .interface import PipelineParams, TimeSeries

DEFAULT_PIPELINE_PARAMS_PATH = Path("hyperparameters/pipeline_hyperparams.yaml")


def load_pipeline_params(path: Path = DEFAULT_PIPELINE_PARAMS_PATH) -> PipelineParams:
    """Load pipeline hyperparameters from YAML. Falls back to PipelineParams defaults if file missing."""
    if not path.exists():
        return PipelineParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return PipelineParams(**data)


class AnomalyPipeline:

    def __init__(
        self,
        model: AnomalyModel,
        pipeline_params: PipelineParams | None = None,
        params_path: Path | None = None,
    ):
        params = pipeline_params or load_pipeline_params(
            params_path or DEFAULT_PIPELINE_PARAMS_PATH
        )
        self.model = model
        self.engine = AlertEngine()
        self.window_size = timedelta(hours=params.model_window_size_hours)
        self.window_overlap = timedelta(hours=params.window_overlap_hours)

    def _windowing_ts(self, ts: TimeSeries) -> List[TimeSeries]:
        if not ts.data:
            return []

        ordered = sorted(ts.data, key=lambda p: p.timestamp)
        first_ts = ordered[0].timestamp
        last_ts = ordered[-1].timestamp

        # Step between consecutive window starts (window_size - overlap); ensure positive to avoid infinite loop
        step = self.window_size - self.window_overlap
        if step <= timedelta(0):
            step = self.window_size

        windows: List[TimeSeries] = []
        window_start = first_ts

        while True:
            window_end = window_start + self.window_size
            points_in_window = [
                p for p in ordered if window_start <= p.timestamp < window_end
            ]
            if not points_in_window:
                break
            windows.append(TimeSeries(data=points_in_window))
            if window_end > last_ts:
                break
            window_start += step

        return windows

    def _predict_windows(self, ts: TimeSeries) -> list[bool]:

        window_predictions = []
        windows = self._windowing_ts(ts)

        for window in windows:
            predict_output = self.model.predict(window)
            window_predictions.append(predict_output)

        return window_predictions

    def predict(self, ts: TimeSeries) -> List[AlertDecision]:
        decisions: List[AlertDecision] = []
        for window in self._windowing_ts(ts):
            pred = self.model.predict(window)
            decisions.append(self.engine.predict(pred))
        return decisions
