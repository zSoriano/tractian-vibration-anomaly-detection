from pathlib import Path

import yaml

from .interface import AlertDecision, AlertEngineParams, PredictOutput

DEFAULT_PARAMS_PATH = Path("hyperparameters/alert_engine_hyperparams.yaml")


def load_alert_engine_params(
    path: Path = DEFAULT_PARAMS_PATH,
) -> AlertEngineParams:
    """Load alert engine hyperparameters from YAML. Falls back to defaults if file missing."""
    if not path.exists():
        return AlertEngineParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AlertEngineParams(**data)


class AlertEngine:
    def __init__(
        self,
        normal_windows_to_unlock: int | None = None,
        anomaly_windows_to_realert: int | None = None,
        severity_increase_factor: float | None = None,
        min_severity_increase: float | None = None,
        severity_windows_to_alert: int | None = None,
        params_path: Path | None = None,
    ):
        params = load_alert_engine_params(params_path or DEFAULT_PARAMS_PATH)

        self.locked = False
        self.has_alerted_once = False
        self.normal_windows_to_unlock = (
            normal_windows_to_unlock
            if normal_windows_to_unlock is not None
            else params.normal_windows_to_unlock
        )
        self.anomaly_windows_to_realert = (
            anomaly_windows_to_realert
            if anomaly_windows_to_realert is not None
            else params.anomaly_windows_to_realert
        )
        self.severity_increase_factor = (
            severity_increase_factor
            if severity_increase_factor is not None
            else params.severity_increase_factor
        )
        self.min_severity_increase = (
            min_severity_increase
            if min_severity_increase is not None
            else params.min_severity_increase
        )
        self.severity_windows_to_alert = (
            severity_windows_to_alert
            if severity_windows_to_alert is not None
            else params.severity_windows_to_alert
        )
        self.normal_window_count = 0
        self.pending_anomaly_count = 0
        self.pending_severity_count = 0
        self.last_alert_severity = 0.0

    def _has_alert(self, prediction: PredictOutput) -> bool:
        return prediction.anomaly_status

    def _reset_recovery(self) -> None:
        self.normal_window_count = 0

    def _reset_pending_anomaly(self) -> None:
        self.pending_anomaly_count = 0

    def _reset_pending_severity(self) -> None:
        self.pending_severity_count = 0

    def _has_severity_increase(self, prediction: PredictOutput) -> bool:
        severity_delta = prediction.severity - self.last_alert_severity
        return (
            prediction.severity
            >= self.last_alert_severity * self.severity_increase_factor
            and severity_delta >= self.min_severity_increase
        )

    def _emit_alert(self, prediction: PredictOutput, message: str) -> AlertDecision:
        self.locked = True
        self.has_alerted_once = True
        self.last_alert_severity = prediction.severity
        self._reset_recovery()
        self._reset_pending_anomaly()
        self._reset_pending_severity()

        return AlertDecision(
            alert=True,
            timestamp=prediction.timestamp,
            message=message,
        )

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        if self.locked:
            self._reset_pending_anomaly()
            if self._has_alert(prediction):
                self._reset_recovery()
                if self._has_severity_increase(prediction):
                    self.pending_severity_count += 1
                    if self.pending_severity_count >= self.severity_windows_to_alert:
                        return self._emit_alert(
                            prediction,
                            "Abnormal vibration severity increased.",
                        )
                else:
                    self._reset_pending_severity()

                return AlertDecision(
                    alert=False,
                    timestamp=prediction.timestamp,
                    message="System remains in abnormal state.",
                )

            self._reset_pending_severity()
            self.normal_window_count += 1
            if self.normal_window_count >= self.normal_windows_to_unlock:
                self.locked = False
                self._reset_recovery()

            return AlertDecision(
                alert=False,
                timestamp=prediction.timestamp,
                message="System recovering from abnormal state.",
            )

        if self._has_alert(prediction):
            if not self.has_alerted_once:
                return self._emit_alert(
                    prediction,
                    "Abnormal vibration detected.",
                )

            self.pending_anomaly_count += 1
            if self.pending_anomaly_count >= self.anomaly_windows_to_realert:
                return self._emit_alert(
                    prediction,
                    "New abnormal vibration event confirmed.",
                )

            return AlertDecision(
                alert=False,
                timestamp=prediction.timestamp,
                message="New anomaly candidate waiting for confirmation.",
            )

        self._reset_pending_anomaly()
        return AlertDecision(
            alert=False,
            timestamp=prediction.timestamp,
            message="No persistent abnormal vibration.",
        )
