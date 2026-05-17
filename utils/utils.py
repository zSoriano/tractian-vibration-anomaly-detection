from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from classes.anomaly_model import AnomalyModel
from classes.data_pipeline import AnomalyPipeline, load_pipeline_params
from classes.interface import (
    AlertDecision,
    DataPoint,
    PipelineParams,
    TimeSeries,
    TrueIncident,
)

# --- Evaluation (moved from classes.evaluate) ---


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def overlap(
    target_start: datetime,
    target_end: datetime,
    prediction: datetime,
) -> bool:
    target_start = ensure_utc(target_start)
    target_end = ensure_utc(target_end)
    prediction = ensure_utc(prediction)
    return target_start <= prediction <= target_end


def extract_alerts(decisions: List[AlertDecision]) -> List[datetime]:
    alerts = []
    for d in decisions:
        if not getattr(d, "alert", False):
            continue
        alerts.append(ensure_utc(d.timestamp))
    return alerts


def seconds_to_timedelta(value: Optional[float]) -> Optional[timedelta]:
    if value is None:
        return None
    return timedelta(seconds=float(value))


def seconds_to_hours(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / 3600.0


def match(
    true_incidents: List[TrueIncident],
    decisions: List[AlertDecision],
) -> Tuple[int, int, int, List[Tuple[datetime, datetime]]]:
    alerts = extract_alerts(decisions)
    matched_alerts: set[int] = set()
    TP = 0
    FN = 0
    for true in true_incidents:
        detected = False
        for i, pe in enumerate(alerts):
            if overlap(true.start, true.end, pe):
                detected = True
                matched_alerts.add(i)
        if detected:
            TP += 1
        else:
            FN += 1
    FP = len(alerts) - len(matched_alerts)
    return TP, FP, FN, alerts


def lead_time(
    true_incidents: List[TrueIncident],
    alerts: List[Tuple[datetime, datetime]],
) -> Optional[float]:
    leads: List[float] = []
    for true in true_incidents:
        duration = (true.end - true.start).total_seconds()
        for ps, pe in alerts:
            if overlap(true.start, true.end, ps, pe):
                delay = (ps - true.start).total_seconds()
                leads.append(duration - delay)
                break
    if not leads:
        return None
    return seconds_to_hours(float(np.mean(leads)))


def evaluate(
    true_incidents: List[TrueIncident],
    decisions: List[AlertDecision],
) -> Dict[str, Any]:
    true_incidents = [
        TrueIncident(start=ensure_utc(t.start), end=ensure_utc(t.end))
        for t in true_incidents
    ]
    TP, FP, FN, alerts = match(true_incidents, decisions)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    metrics = {
        "n_true_incidents": len(true_incidents),
        "n_predicted_alerts": len(alerts),
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TPR": TP / len(true_incidents) if len(true_incidents) > 0 else np.nan,
        "FPR": FP / len(alerts) if len(alerts) > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


def _avg(values: List[Optional[float]]) -> Optional[float]:
    values = [abs(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    runs = list(results.values())
    TP = sum(r["TP"] for r in runs)
    FP = sum(r["FP"] for r in runs)
    FN = sum(r["FN"] for r in runs)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "runs": len(runs),
        "TP_total": TP,
        "FP_total": FP,
        "FN_total": FN,
        "precision_global": precision,
        "recall_global": recall,
        "f1_global": f1,
    }


# --- Data / timeseries ---


def df_to_timeseries(df: pd.DataFrame) -> TimeSeries:
    if df.empty:
        return TimeSeries(data=[])

    df = df.sort_values("sampled_at")
    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True)

    datapoints: List[DataPoint] = []

    for row in df.itertuples(index=False):

        datapoints.append(
            DataPoint(
                timestamp=row.sampled_at,
                uptime=row.uptime,
                vel_x=float(row.vel_rms_x),
                vel_y=float(row.vel_rms_y),
                vel_z=float(row.vel_rms_z),
                acc_x=float(row.accel_rms_x),
                acc_y=float(row.accel_rms_y),
                acc_z=float(row.accel_rms_z),
            )
        )

    return TimeSeries(data=datapoints)


def split_timeseries(
    ts: TimeSeries, fit_ratio: float = 0.3
) -> Tuple[TimeSeries, TimeSeries]:
    if not ts.data:
        return TimeSeries(data=[]), TimeSeries(data=[])

    ordered = sorted(ts.data, key=lambda p: p.timestamp)

    split_index = int(len(ordered) * fit_ratio)

    fit_data = ordered[:split_index]
    predict_data = ordered[split_index:]

    return TimeSeries(data=fit_data), TimeSeries(data=predict_data)


def run_experiment(
    fit_ts: TimeSeries,
    predict_ts: TimeSeries,
    true_incidents: List[TrueIncident],
    pipeline_params: PipelineParams | None = None,
) -> Tuple[Dict[str, Any], List[AlertDecision]]:
    model = AnomalyModel()
    model.fit(fit_ts)

    if pipeline_params is None:
        pipeline_params = load_pipeline_params()
    pipeline = AnomalyPipeline(model, pipeline_params=pipeline_params)
    predictions = pipeline.predict(predict_ts)
    metrics = evaluate(true_incidents, predictions)

    return metrics, predictions


def timeseries_to_df(ts: TimeSeries) -> pd.DataFrame:
    """
    Convert TimeSeries -> pandas DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for p in ts.data:

        row = {
            "sampled_at": p.timestamp,
            "uptime": p.uptime,
            "vel_rms_x": getattr(p, "vel_x", None),
            "vel_rms_y": getattr(p, "vel_y", None),
            "vel_rms_z": getattr(p, "vel_z", None),
            "accel_rms_x": getattr(p, "acc_x", None),
            "accel_rms_y": getattr(p, "acc_y", None),
            "accel_rms_z": getattr(p, "acc_z", None),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True)

    df = df.sort_values("sampled_at").reset_index(drop=True)

    return df


def plot_sensor_with_incidents(
    ts: TimeSeries,
    true_incidents: List[TrueIncident],
    decisions: Optional[List[AlertDecision]] = None,
    title: Optional[str] = None,
    return_figure: bool = False,
) -> Optional[go.Figure]:
    df = timeseries_to_df(ts)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Velocity RMS", "Acceleration RMS"),
    )

    # Velocity RMS (row 1)
    fig.add_trace(
        go.Scatter(
            x=df["sampled_at"],
            y=df["vel_rms_x"],
            name="vel_x",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["sampled_at"],
            y=df["vel_rms_y"],
            name="vel_y",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["sampled_at"],
            y=df["vel_rms_z"],
            name="vel_z",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )

    # Acceleration RMS (row 2)
    fig.add_trace(
        go.Scatter(
            x=df["sampled_at"],
            y=df["accel_rms_x"],
            name="acc_x",
            line=dict(width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["sampled_at"],
            y=df["accel_rms_y"],
            name="acc_y",
            line=dict(width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["sampled_at"],
            y=df["accel_rms_z"],
            name="acc_z",
            line=dict(width=1),
        ),
        row=2,
        col=1,
    )

    vel_min = min(df["vel_rms_x"].min(), df["vel_rms_y"].min(), df["vel_rms_z"].min())
    vel_max = max(df["vel_rms_x"].max(), df["vel_rms_y"].max(), df["vel_rms_z"].max())
    acc_min = min(
        df["accel_rms_x"].min(),
        df["accel_rms_y"].min(),
        df["accel_rms_z"].min(),
    )
    acc_max = max(
        df["accel_rms_x"].max(),
        df["accel_rms_y"].max(),
        df["accel_rms_z"].max(),
    )

    # True incidents as Scatter traces (so legend toggle shows/hides them)
    first_incident_legend = True
    for inc in true_incidents:
        start = pd.to_datetime(inc.start, utc=True)
        end = pd.to_datetime(inc.end, utc=True)
        fig.add_trace(
            go.Scatter(
                x=[start, start, end, end, start],
                y=[vel_min, vel_max, vel_max, vel_min, vel_min],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(255,0,0,0.25)",
                name="True Incident",
                legendgroup="true_incident",
                showlegend=first_incident_legend,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[start, start, end, end, start],
                y=[acc_min, acc_max, acc_max, acc_min, acc_min],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(255,0,0,0.25)",
                name="True Incident",
                legendgroup="true_incident",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        first_incident_legend = False

    first_alert_legend = True
    if decisions is not None:
        for d in decisions:
            if not d.alert:
                continue
            end_ts = pd.to_datetime(d.timestamp, utc=True)
            fig.add_trace(
                go.Scatter(
                    x=[end_ts, end_ts],
                    y=[vel_min, vel_max],
                    mode="lines",
                    line=dict(color="blue", width=2, dash="dash"),
                    name="Pipeline Alert",
                    legendgroup="pipeline_alert",
                    showlegend=first_alert_legend,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[end_ts, end_ts],
                    y=[acc_min, acc_max],
                    mode="lines",
                    line=dict(color="blue", width=2, dash="dash"),
                    name="Pipeline Alert",
                    legendgroup="pipeline_alert",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            first_alert_legend = False

    layout_kw: Dict[str, Any] = dict(
        height=500,
        margin=dict(t=40, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        xaxis=dict(tickangle=-45),
    )
    if title is not None:
        layout_kw["title"] = title
    fig.update_layout(**layout_kw)
    fig.update_yaxes(title_text="mm/s", row=1, col=1)
    fig.update_yaxes(title_text="g", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    if return_figure:
        return fig
    fig.show()
