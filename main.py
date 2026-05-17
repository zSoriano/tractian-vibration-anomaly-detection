#!/usr/bin/env python3
"""
Run the anomaly detection experiment across all scenarios.

Usage (from tests/tech_program):
    python main.py
    python main.py --no-plot --no-save
    python main.py --output-dir ./out
"""
import argparse
from pathlib import Path

from utils.data_loading import load_all_scenarios, save_metrics
from utils.utils import aggregate_results, plot_sensor_with_incidents, run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Run anomaly detection experiment")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiment_outputs"),
        help="Directory for metrics JSON outputs",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save metrics to JSON",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not plot predictions per scenario",
    )
    return p.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_flag = not args.no_save
    plot_predictions = not args.no_plot

    results = {}
    all_predictions = {}
    scenarios = load_all_scenarios()

    for i, (fit_ts, predict_ts, true_incidents) in enumerate(scenarios, start=1):
        metrics, preds = run_experiment(
            fit_ts,
            predict_ts,
            true_incidents,
        )

        results[f"scenario_{i}"] = metrics
        all_predictions[f"scenario_{i}"] = preds

        if save_metrics_flag:
            save_metrics(metrics, args.output_dir / f"metrics_scenario_{i}.json")

        if plot_predictions:
            plot_sensor_with_incidents(
                predict_ts,
                true_incidents,
                decisions=all_predictions[f"scenario_{i}"],
                title=f"scenario_{i}",
            )

    general_metric = aggregate_results(results)

    if save_metrics_flag:
        save_metrics(
            general_metric,
            args.output_dir / "metrics_all_scenarios.json",
        )

    return results, general_metric


if __name__ == "__main__":
    main()
