from datetime import datetime
from typing import Sequence, List

from pydantic import BaseModel, Field


# --- Domain / data transfer types ---


class DataPoint(BaseModel):
    timestamp: datetime = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    uptime: bool = Field(..., description="Whether the data point is during uptime")
    vel_x: float = Field(
        ..., description="Vibration velocity component along the X axis"
    )
    vel_y: float = Field(
        ..., description="Vibration velocity component along the Y axis"
    )
    vel_z: float = Field(
        ..., description="Vibration velocity component along the Z axis"
    )

    acc_x: float = Field(
        ..., description="Vibration acceleration component along the X axis"
    )
    acc_y: float = Field(
        ..., description="Vibration acceleration component along the Y axis"
    )
    acc_z: float = Field(
        ..., description="Vibration acceleration component along the Z axis"
    )


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def last_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[-1].timestamp

    @property
    def first_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[0].timestamp


# --- Model / pipeline types (shared across modules) ---


class Weights(BaseModel):
    fitted: bool = False
    feature_names: List[str] = Field(default_factory=list)
    center: List[float] = Field(default_factory=list)
    scale: List[float] = Field(default_factory=list)


class ModelParams(BaseModel):
    velocity_z_threshold: float = 4.0
    acceleration_z_threshold: float = 5.0
    velocity_window_anomaly_ratio: float = 0.2
    acceleration_window_anomaly_ratio: float = 0.3
    min_scale: float = 1e-6


class PipelineParams(BaseModel):
    model_window_size_hours: float = 4.0
    window_overlap_hours: float = 0.0


class AlertEngineParams(BaseModel):
    normal_windows_to_unlock: int = 12
    anomaly_windows_to_realert: int = 3
    severity_increase_factor: float = 3.0
    min_severity_increase: float = 1.0
    severity_windows_to_alert: int = 3


class PredictOutput(BaseModel):
    anomaly_status: bool
    timestamp: datetime
    severity: float = 0.0


class AlertDecision(BaseModel):
    alert: bool
    timestamp: datetime
    message: str


class TrueIncident(BaseModel):
    start: datetime
    end: datetime
