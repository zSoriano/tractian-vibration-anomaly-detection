from pathlib import Path

import numpy as np
import yaml

from .interface import ModelParams, PredictOutput, TimeSeries, Weights

DEFAULT_PARAMS_PATH = Path("hyperparameters/model_hyperparams.yaml")

FEATURE_NAMES = (
    "vel_norm",
    "acc_norm",
    "vel_axis_max",
    "acc_axis_max",
)

VELOCITY_FEATURE_INDICES = (
    FEATURE_NAMES.index("vel_norm"),
    FEATURE_NAMES.index("vel_axis_max"),
)
ACCELERATION_FEATURE_INDICES = (
    FEATURE_NAMES.index("acc_norm"),
    FEATURE_NAMES.index("acc_axis_max"),
)


def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to ModelParams defaults if file missing."""
    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelParams(**data)


class AnomalyModel:
    def __init__(self, params_path: Path | None = None):
        self.weights = Weights()
        self.params = load_model_params(params_path or DEFAULT_PARAMS_PATH)

    def _featuring(self, samples: TimeSeries) -> np.ndarray:
        if not samples.data:
            return np.empty((0, len(FEATURE_NAMES)), dtype=float)

        ordered = sorted(samples.data, key=lambda p: p.timestamp)

        rows = []
        for p in ordered:
            vel_values = [p.vel_x, p.vel_y, p.vel_z]
            acc_values = [p.acc_x, p.acc_y, p.acc_z]
            vel_abs_values = [abs(value) for value in vel_values]
            acc_abs_values = [abs(value) for value in acc_values]
            vel_norm = np.linalg.norm(vel_values)
            acc_norm = np.linalg.norm(acc_values)
            vel_axis_max = max(vel_abs_values)
            acc_axis_max = max(acc_abs_values)

            rows.append(
                [
                    vel_norm,
                    acc_norm,
                    vel_axis_max,
                    acc_axis_max,
                ]
            )

        return np.array(rows, dtype=float)

    def fit(self, fitting_samples: TimeSeries) -> None:
        X = self._featuring(fitting_samples)

        if X.size == 0:
            raise ValueError("Cannot fit on empty TimeSeries")

        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > self.params.min_scale, scale, self.params.min_scale)

        self.weights = Weights(
            fitted=True,
            feature_names=list(FEATURE_NAMES),
            center=np.round(center, 6).tolist(),
            scale=np.round(scale, 6).tolist(),
        )

    def _zscore(self, X: np.ndarray) -> np.ndarray:
        center = np.array(self.weights.center, dtype=float)
        scale = np.array(self.weights.scale, dtype=float)

        return np.abs((X - center) / scale)

    def _window_anomaly_scores(self, X: np.ndarray) -> tuple[float, float, float]:
        z = self._zscore(X)

        velocity_scores = np.max(z[:, VELOCITY_FEATURE_INDICES], axis=1)
        acceleration_scores = np.max(z[:, ACCELERATION_FEATURE_INDICES], axis=1)

        velocity_ratio = np.mean(
            velocity_scores > self.params.velocity_z_threshold
        )
        acceleration_ratio = np.mean(
            acceleration_scores > self.params.acceleration_z_threshold
        )

        velocity_severity = (
            np.percentile(velocity_scores, 95) / self.params.velocity_z_threshold
        )
        acceleration_severity = (
            np.percentile(acceleration_scores, 95)
            / self.params.acceleration_z_threshold
        )
        severity = max(velocity_severity, acceleration_severity)

        return float(velocity_ratio), float(acceleration_ratio), float(severity)

    def predict(self, samples: TimeSeries) -> PredictOutput:
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        X = self._featuring(samples)
        velocity_ratio, acceleration_ratio, severity = self._window_anomaly_scores(X)
        is_anomalous = (
            velocity_ratio >= self.params.velocity_window_anomaly_ratio
            or acceleration_ratio >= self.params.acceleration_window_anomaly_ratio
        )

        return PredictOutput(
            anomaly_status=is_anomalous,
            timestamp=samples.data[-1].timestamp,
            severity=severity,
        )
