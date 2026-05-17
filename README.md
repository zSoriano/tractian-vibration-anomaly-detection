# Vibration Anomaly Detection — Technical Test

## Context

Industrial machines are continuously monitored by vibration sensors. These sensors capture acceleration and velocity readings across three axes (X, Y, Z) at regular intervals. When a machine starts to fail, the vibration pattern changes — and the goal of this system is to detect that change early and raise an alert.

The system has three core product requirements:

1. **Detect anomalous high-vibration patterns** — when a machine's vibration deviates significantly from its normal operating baseline, an alert should be raised.
2. **Avoid repetitive alerts** — once an abnormal state has been flagged, the system should not keep firing alerts for the same ongoing event.
3. **Re-alert on escalation** — if the vibration condition worsens after a prior alert has been issued, a new alert should be raised to signal the aggravated state.

You are given a working but intentionally flawed anomaly detection pipeline. Your task is to understand the system, identify its weaknesses, and improve it.

---

## The data

The dataset is provided as `data.zip`. Extract its contents into the `data/` folder before running anything:

```bash
unzip data.zip -d data/
```

The `data/` folder contains parquet files covering multiple independent scenarios, each representing a different physical asset (machine). Files are split by purpose:

- `vibe_data_fit_{i}.parquet` — historical data used to calibrate the model (what "normal" looks like for this asset)
- `vibe_data_pred_{i}.parquet` — data the model runs inference on

Each file has the following columns:


| Column        | Description                                           |
| ------------- | ----------------------------------------------------- |
| `sampled_at`  | UTC timestamp of the measurement                      |
| `uptime`      | Boolean — whether the machine was running at the time |
| `vel_rms_x`   | Velocity RMS along X axis (mm/s)                      |
| `vel_rms_y`   | Velocity RMS along Y axis (mm/s)                      |
| `vel_rms_z`   | Velocity RMS along Z axis (mm/s)                      |
| `accel_rms_x` | Acceleration RMS along X axis (g)                     |
| `accel_rms_y` | Acceleration RMS along Y axis (g)                     |
| `accel_rms_z` | Acceleration RMS along Z axis (g)                     |


Measurements are taken roughly every 10 minutes. Prediction periods vary in length depending on the scenario.

---

## Ground truth

`labels/incidents.yaml` contains the true incident windows for each scenario — the time ranges during which the machine was genuinely in an abnormal state. Some scenarios have a single incident; others have two.

A predicted alert **counts as a true positive** if its timestamp falls within a true incident window.

---

## The pipeline

The system runs as follows:

```
fit data  ──►  AnomalyModel.fit()   learns a baseline for this asset
                      │
pred data ──►  sliding windows  ──►  AnomalyModel.predict()  ──►  anomaly_status (bool)
                                              │
                                       AlertEngine.predict()  ──►  AlertDecision (alert bool)
```

**Windowing:** the prediction time series is sliced into overlapping fixed-size windows. Each window is scored independently by the model.

**Alert engine:** receives one `PredictOutput` per window and immediately returns an `AlertDecision`.

Results are written to `experiment_outputs/` as JSON files (one per scenario, plus an aggregate).

---

## How to run

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Run the full experiment (fits, predicts, saves metrics, shows plots)
uv run main.py

# Run without plots
uv run main.py --no-plot

# Run without saving output files
uv run main.py --no-save
```

The outputs land in `experiment_outputs/`:

- `metrics_scenario_{i}.json` — per-scenario TP, FP, FN, precision, recall, F1
- `metrics_all_scenarios.json` — aggregated metrics across all scenarios

---

## Baseline performance

The current implementation produces intentionally poor results. Your goal is to improve them.

---

## What you can change


| File                       | Can modify? | Notes                                                                                                                |
| -------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `classes/anomaly_model.py` | **Yes**     | `fit`, `predict`, and any private helpers                                                                            |
| `classes/alert_engine.py`  | **Yes**     | Full class                                                                                                           |
| `classes/interface.py`     | **Yes**     | Add new classes or add parameters to existing ones                                                                   |
| `hyperparameters/*.yaml`   | **Yes**     | Add new files or change existing values                                                                              |
| `classes/data_pipeline.py` | No          |                                                                                                                      |
| `utils/`                   | No          |                                                                                                                      |
| `main.py`                  | No          |                                                                                                                      |


The pipeline wiring, evaluation logic, and data loading are fixed.

---

## What we are looking for

**Reasoning over results.** We care more about *why* you made each decision than the final metric alone.

**Clean implementation.** Code quality is a must. Your code should be readable and fit naturally into the existing structure.

---

## Deliverables

1. **A GitHub repository** — share the link to your solution. It should include all modified source files and the report.
2. `REPORT.md` — a write-up covering:
  - What issues you identified in the baseline
  - What changes you made and why
  - The final metrics you achieved
  - Any limitations or next steps you would explore given more time

