import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from classes.interface import TimeSeries, TrueIncident
from utils.utils import df_to_timeseries

DATA_DIR = Path("./data")
LABELS_DIR = Path("./labels")
INCIDENTS_FILE = LABELS_DIR / "incidents.yaml"


def _load_incidents(incidents_path: Path = INCIDENTS_FILE) -> Dict[int, List[TrueIncident]]:
    """Load incident labels from YAML. Keys are scenario IDs, values are lists of TrueIncident."""
    if not incidents_path.exists():
        return {}
    with open(incidents_path) as f:
        raw = yaml.safe_load(f) or {}
    result: Dict[int, List[TrueIncident]] = {}
    for scenario_key, intervals in raw.items():
        scenario_id = int(scenario_key)
        result[scenario_id] = []
        for item in intervals:
            start = datetime.fromisoformat(item["start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(item["end"].replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            result[scenario_id].append(TrueIncident(start=start, end=end))
    return result


def load_all_scenarios(
    data_dir: Path = DATA_DIR,
    incidents_path: Path = INCIDENTS_FILE,
) -> List[Tuple[TimeSeries, TimeSeries, List[TrueIncident]]]:
    incidents = _load_incidents(incidents_path)
    scenarios = []

    for i in sorted(incidents.keys()):

        fit_path = data_dir / f"vibe_data_fit_{i}.parquet"
        pred_path = data_dir / f"vibe_data_pred_{i}.parquet"

        if not fit_path.exists() or not pred_path.exists():
            print(f"[WARN] Missing files for scenario {i}")
            continue
        df_fit = pd.read_parquet(fit_path)
        df_pred = pd.read_parquet(pred_path)

        fit_ts = df_to_timeseries(df_fit)
        pred_ts = df_to_timeseries(df_pred)

        true_incidents = incidents[i]

        scenarios.append((fit_ts, pred_ts, true_incidents))

    return scenarios


def save_metrics(metrics: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f)
