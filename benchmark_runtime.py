#!/usr/bin/env python3
"""
Benchmark runtime scaling for the anomaly detection pipeline.

This script is intentionally separate from the main experiment pipeline. It reuses
the existing public classes and measures how runtime changes as each scenario is
evaluated with increasing fractions of the available data.

Usage:
    python benchmark_runtime.py
    python benchmark_runtime.py --fractions 0.25 0.50 1.00 --repetitions 3
"""

import argparse
import csv
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from classes.alert_engine import AlertEngine
from classes.anomaly_model import AnomalyModel
from classes.data_pipeline import AnomalyPipeline, load_pipeline_params
from classes.interface import TimeSeries
from utils.data_loading import load_all_scenarios

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = Path("experiment_outputs/runtime_benchmark_by_scenario.csv")
DEFAULT_SUMMARY_PATH = Path("experiment_outputs/runtime_benchmark_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure runtime scaling by scenario and input size."
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
        help="Fractions of each scenario to benchmark.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per scenario/fraction. Median time is reported.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Detailed CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Summary CSV output path grouped by input fraction.",
    )
    return parser.parse_args()


def validate_fractions(fractions: list[float]) -> list[float]:
    unique_fractions = sorted(set(fractions))
    invalid = [value for value in unique_fractions if value <= 0 or value > 1]
    if invalid:
        raise ValueError(f"Fractions must be greater than 0 and up to 1: {invalid}")
    return unique_fractions


def slice_timeseries(ts: TimeSeries, fraction: float) -> TimeSeries:
    if not ts.data:
        return TimeSeries(data=[])

    ordered = sorted(ts.data, key=lambda point: point.timestamp)
    size = max(1, int(len(ordered) * fraction))
    return TimeSeries(data=ordered[:size])


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def round_seconds(value: float) -> float:
    return round(float(value), 6)


def measure_once(fit_ts: TimeSeries, predict_ts: TimeSeries) -> dict[str, Any]:
    pipeline_params = load_pipeline_params()

    model = AnomalyModel()
    start = perf_counter()
    model.fit(fit_ts)
    fit_seconds = perf_counter() - start

    pipeline = AnomalyPipeline(model, pipeline_params=pipeline_params)

    start = perf_counter()
    windows = pipeline._windowing_ts(predict_ts)
    windowing_seconds = perf_counter() - start

    start = perf_counter()
    predictions = [model.predict(window) for window in windows]
    predict_seconds = perf_counter() - start

    engine = AlertEngine()
    start = perf_counter()
    decisions = [engine.predict(prediction) for prediction in predictions]
    alert_engine_seconds = perf_counter() - start

    return {
        "n_windows": len(windows),
        "n_anomaly_windows": sum(
            1 for prediction in predictions if prediction.anomaly_status
        ),
        "n_alerts": sum(1 for decision in decisions if decision.alert),
        "fit_seconds": fit_seconds,
        "windowing_seconds": windowing_seconds,
        "predict_seconds": predict_seconds,
        "alert_engine_seconds": alert_engine_seconds,
        "total_seconds": (
            fit_seconds
            + windowing_seconds
            + predict_seconds
            + alert_engine_seconds
        ),
    }


def benchmark_scenario(
    scenario_id: int,
    fit_ts: TimeSeries,
    predict_ts: TimeSeries,
    fraction: float,
    repetitions: int,
) -> dict[str, Any]:
    fit_slice = slice_timeseries(fit_ts, fraction)
    predict_slice = slice_timeseries(predict_ts, fraction)
    measurements = [measure_once(fit_slice, predict_slice) for _ in range(repetitions)]

    return {
        "scenario": scenario_id,
        "fraction": fraction,
        "fit_rows": fit_slice.length,
        "predict_rows": predict_slice.length,
        "n_windows": int(median([row["n_windows"] for row in measurements])),
        "n_anomaly_windows": int(
            median([row["n_anomaly_windows"] for row in measurements])
        ),
        "n_alerts": int(median([row["n_alerts"] for row in measurements])),
        "fit_seconds": round_seconds(
            median([row["fit_seconds"] for row in measurements])
        ),
        "windowing_seconds": round_seconds(
            median([row["windowing_seconds"] for row in measurements])
        ),
        "predict_seconds": round_seconds(
            median([row["predict_seconds"] for row in measurements])
        ),
        "alert_engine_seconds": round_seconds(
            median([row["alert_engine_seconds"] for row in measurements])
        ),
        "total_seconds": round_seconds(
            median([row["total_seconds"] for row in measurements])
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    fractions = sorted({row["fraction"] for row in rows})

    for fraction in fractions:
        fraction_rows = [row for row in rows if row["fraction"] == fraction]
        summary_rows.append(
            {
                "fraction": fraction,
                "scenarios": len(fraction_rows),
                "fit_rows_total": sum(row["fit_rows"] for row in fraction_rows),
                "predict_rows_total": sum(row["predict_rows"] for row in fraction_rows),
                "n_windows_total": sum(row["n_windows"] for row in fraction_rows),
                "fit_seconds_total": round_seconds(
                    sum(row["fit_seconds"] for row in fraction_rows)
                ),
                "windowing_seconds_total": round_seconds(
                    sum(row["windowing_seconds"] for row in fraction_rows)
                ),
                "predict_seconds_total": round_seconds(
                    sum(row["predict_seconds"] for row in fraction_rows)
                ),
                "alert_engine_seconds_total": round_seconds(
                    sum(row["alert_engine_seconds"] for row in fraction_rows)
                ),
                "total_seconds_total": round_seconds(
                    sum(row["total_seconds"] for row in fraction_rows)
                ),
            }
        )

    return summary_rows


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nRuntime summary by input fraction")
    print("fraction | predict_rows | windows | total_seconds")
    print("-" * 52)
    for row in summary_rows:
        print(
            f"{row['fraction']:.2f}     | "
            f"{row['predict_rows_total']:>12} | "
            f"{row['n_windows_total']:>7} | "
            f"{row['total_seconds_total']:.6f}"
        )


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    fractions = validate_fractions(args.fractions)
    repetitions = max(1, args.repetitions)

    scenarios = load_all_scenarios()
    if not scenarios:
        raise RuntimeError("No scenarios were loaded. Check the data and labels paths.")

    rows = []
    for scenario_id, (fit_ts, predict_ts, _) in enumerate(scenarios, start=1):
        for fraction in fractions:
            rows.append(
                benchmark_scenario(
                    scenario_id=scenario_id,
                    fit_ts=fit_ts,
                    predict_ts=predict_ts,
                    fraction=fraction,
                    repetitions=repetitions,
                )
            )

    write_csv(args.output, rows)
    summary_rows = build_summary(rows)
    write_csv(args.summary_output, summary_rows)
    print_summary(summary_rows)
    print(f"\nDetailed table saved to: {args.output}")
    print(f"Summary table saved to: {args.summary_output}")


if __name__ == "__main__":
    main()
