# Vibration Anomaly Detection Report

## Baseline Analysis

The original implementation used a single vibration feature: the Euclidean norm of the three velocity axes. The model fitted one global mean and one global standard deviation for that feature and marked a window as anomalous when a sufficient share of points exceeded a fixed z-score threshold. This approach ignored acceleration and lost information about directional changes across the axes.

The initial alert logic also had a structural problem. After the first anomalous window, the `AlertEngine` entered a permanent locked state and never emitted a second alert. That behavior avoided repetitive alerts, but it prevented the system from detecting later incidents or a renewed abnormal state after recovery.

The original aggregate baseline metrics were:

```text
TP=5
FP=14
FN=24
precision=0.2632
recall=0.1724
f1=0.2083
```

During the analysis, four failure patterns became clear. The first was the absence of acceleration as a feature: some incidents were more visible in acceleration than in velocity, but the baseline ignored acceleration completely. The second was the alert lock blocking later detections: scenarios with more than one incident were strongly affected because the permanent lock after the first alert turned later incidents into structural false negatives. The third was a double-penalty effect caused by early alerts: in some scenarios, the model alerted slightly before the labeled incident window, generating a false positive outside the incident and a false negative inside it because the lock prevented a new alert. The fourth was a data issue: one labeled interval in scenario 7 has a start timestamp later than its end timestamp, which the provided evaluator cannot match. This false negative is structural and independent of detector quality. Since `main.py` is immutable, that invalid interval remains in the final metric calculation and should be considered when interpreting the results.

## Methodology

### Feature Engineering

The model was updated to use velocity and acceleration while keeping the feature set compact and explainable. Using every raw axis as an independent feature would increase sensitivity to isolated noise in a single axis without adding much useful information. The final model uses four derived features, described below.

| Feature | Description |
|---|---|
| `vel_norm` | Euclidean norm of the three velocity axes |
| `acc_norm` | Euclidean norm of the three acceleration axes |
| `vel_axis_max` | Maximum absolute value among the three velocity axes |
| `acc_axis_max` | Maximum absolute value among the three acceleration axes |

The norm captures total vibration energy. The dominant-axis feature captures directional abnormal behavior. These two perspectives are complementary and cover patterns the baseline missed without introducing unnecessary axis-level redundancy.

The fit stage still uses mean and standard deviation. Statistics based on median and IQR were tested, but in this dataset they made some thresholds too permissive or too restrictive depending on the scenario. The mean and standard deviation version produced better timing and more stable aggregate metrics after acceleration and dominant-axis features were added. The `min_scale` parameter (`1e-6`) prevents features with near-zero variance from generating artificially high z-scores.

### Independent Physical Groups

Prediction is evaluated separately for velocity and acceleration. For each point in a window, the model computes z-scores for the relevant features. The velocity score is the maximum z-score between `vel_norm` and `vel_axis_max`; the acceleration score is the maximum z-score between `acc_norm` and `acc_axis_max`. A window is anomalous if either group has enough anomalous points. Keeping these physical quantities independently calibrated allows an incident to be detected even when only one of them shows a significant deviation.

### Window Parameters

The pipeline window parameters were kept aligned with the challenge description: 4-hour windows with 2-hour overlap. A local search showed that nearby values could also work, but the 4h/2h setting is directly supported by the product description and reduces the risk of overfitting to the available scenarios.

### Hyperparameter Tuning Process

The hyperparameters were tuned iteratively, scenario by scenario, by reviewing the `metrics_scenario_{i}.json` outputs after each run. The process happened in three stages.

In the first stage, the z-score thresholds started at conservative values, 3.0 for both physical quantities, and the `anomaly_ratio` started at 0.3. This established a low-recall but acceptable-precision baseline for the next adjustments.

In the second stage, thresholds were tuned by physical group. The velocity threshold was increased to 4.5 after observing that low-amplitude scenarios produced consistent false positives in velocity. The acceleration threshold was set to 4.0 because acceleration showed stronger deviations in real incidents and tolerated a slightly lower threshold without losing precision.

In the third stage, the `anomaly_ratio` was reduced from 0.3 to 0.2. This improved recall without degrading precision because real incidents tended to produce persistent anomalous points inside the window, and the lower ratio captured that pattern with more sensitivity.

The final model hyperparameters are:

```yaml
velocity_z_threshold: 4.5
acceleration_z_threshold: 4.0
velocity_window_anomaly_ratio: 0.2
acceleration_window_anomaly_ratio: 0.2
min_scale: 1.0e-6
```

### Alert Engine With Recovery And Severity

The alert engine was changed from a permanent lock to a state machine with recovery and severity tracking. The first anomaly still triggers an immediate alert. Later anomalous windows are suppressed while the abnormal state is active. The engine only unlocks after 12 consecutive normal windows, representing sustained asset recovery. After that recovery period, a new alert requires 3 consecutive anomalous windows, which reduces the chance of re-alerting on transient noise.

The state transitions are:

```text
[NORMAL] -> anomalous window -> [ANOMALOUS: alert emitted]
[ANOMALOUS] -> subsequent anomalous windows -> suppressed
[ANOMALOUS] -> 12 consecutive normal windows -> [RECOVERED/NORMAL]
[NORMAL after recovery] -> 3 consecutive anomalous windows -> [ANOMALOUS: new alert]
[ANOMALOUS] -> severity >= 3x last severity and delta >= 1.0 for 3 windows -> escalation alert
```

The severity score was inspired by the scalar health indicator described by Singh et al. (2019) and the broader anomaly detection discussion by Kamat and Sugandhi (2020). It uses the 95th percentile of velocity and acceleration z-scores normalized by their respective thresholds. The result is a dimensionless value that is comparable across physical groups and traceable over time, which allows the system to identify worsening inside the same abnormal event without depending on a new absolute threshold.

The alert parameters are loaded from `hyperparameters/alert_engine_hyperparams.yaml`, following the same configuration pattern used by the model and pipeline hyperparameters.

### Code Design Decisions

Three implementation decisions are worth noting. The `AlertEngine` hyperparameters were moved to their own YAML file instead of remaining hardcoded in the class, which keeps the implementation consistent with the project pattern and makes configuration comparisons possible without editing code. The alert state is kept explicit through readable state attributes such as lock status, recovery count, pending anomaly count, and pending severity count. The `severity` field was added to `PredictOutput` without changing the external contract of `AnomalyModel.predict()`, so the `AlertEngine` can consume that information without direct coupling to the model internals.

## Performance

### Metric Evolution

| Version | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original baseline | 5 | 14 | 24 | 0.2632 | 0.1724 | 0.2083 |
| New features + AlertEngine | 15 | 5 | 14 | 0.7500 | 0.5172 | 0.6122 |
| Severity model added | 16 | 5 | 13 | 0.7619 | 0.5517 | **0.6400** |

### Scenario-Level Analysis

The table below summarizes the main diagnostic scenarios reviewed during the final analysis, including notes about the remaining errors. Full per-scenario metrics are available in the generated experiment outputs.

| Scenario | TP | FP | FN | Note |
|---|---:|---:|---:|---|
| 1 | 1 | 0 | 0 | Clean detection, single incident |
| 2 | 2 | 0 | 1 | Early timing in one incident; FN due to persistence threshold |
| 3 | 1 | 1 | 0 | FP likely represents anomalous behavior outside the labeled window |
| 4 | 2 | 0 | 2 | Multiple incidents; cooldown is not enough to separate nearby events |
| 5 | 3 | 1 | 2 | Low-amplitude velocity incidents; acceleration was the main signal |
| 6 | 2 | 1 | 2 | FP before the labeled window; detector appears to anticipate the event |
| 7 | 2 | 1 | 3 | 1 structural FN caused by invalid label (`start > end`); remaining FNs due to persistence threshold |
| 8 | 1 | 0 | 1 | Short incident only partially captured by the 4h window |
| 9 | 2 | 1 | 2 | Mixed velocity/acceleration pattern; FP during uptime transition |

The scenario 7 false negative related to the label with `start > end` is structural: the evaluator provided in `main.py` cannot match that interval by design. Excluding that case, the model would have 12 effective false negatives.

The gains over the baseline came from adding acceleration and dominant-axis features, using 4h windows with 2h overlap, tuning thresholds separately by physical group, replacing the permanent lock with a recovery-aware alert engine, and adding the severity model for worsening events.

The remaining false negatives are concentrated in two patterns: short incidents that do not accumulate enough anomalous windows to exceed the persistence threshold, and incidents in multi-event scenarios where the 12-window recovery requirement does not separate nearby events cleanly.

## Complexity Analysis and Runtime Benchmark

The final solution was also reviewed from a Big O perspective. The model fit stage is expected to scale approximately linearly with the number of baseline samples, since it computes feature means and standard deviations. Window-level prediction is also linear in the number of points inside each window. Because the configured window length is fixed at 4 hours, the practical cost of one prediction remains bounded by the window size. The alert engine is constant time per window because it only updates a small set of state counters.

A separate benchmark script, `benchmark_runtime.py`, was added to empirically validate this complexity analysis without changing the main experiment pipeline. The script loads the same scenarios, uses the current model and alert engine, and measures execution time for increasing fractions of the input data. The goal was to confirm whether the implemented approach remains computationally simple and to identify which stage dominates runtime in practice.

The benchmark output is saved in `experiment_outputs/runtime_benchmark_summary.csv`. The latest run produced the following summary:

| Input fraction | Prediction rows | Windows | Total time (s) |
|---:|---:|---:|---:|
| 0.25 | 22,182 | 1,665 | 0.8689 |
| 0.50 | 44,378 | 3,338 | 2.0904 |
| 0.75 | 66,566 | 4,956 | 3.6828 |
| 1.00 | 88,769 | 6,574 | 5.8145 |

The model fit stage and the alert engine were lightweight. At full input size, fitting took approximately 0.38 seconds and alert processing took approximately 0.01 seconds. Most of the runtime came from window construction and window-level prediction. This is consistent with the project design: the model uses simple feature extraction, z-scores, and threshold rules, while the immutable pipeline constructs overlapping sliding windows before calling `AnomalyModel.predict()`.

The benchmark also shows that the number of windows grows proportionally with the number of rows. Most runtime comes from window construction inside the immutable pipeline. Since that stage could not be modified, this is documented as an engineering finding: a pointer-based or indexed time-series approach would avoid repeated scans over overlapping windows and would be the natural next step in a production context. This optimization is carried forward in the Future Work section.

## Future Work

With more data, the next step would be leave-one-scenario-out cross-validation to estimate parameter stability and reduce the risk of overfitting to this specific dataset.

The severity model would also benefit from more labeled examples. The current implementation uses a direct z-score based rule, which is explainable and improved the aggregate metric, but a dataset with labeled escalation examples would allow the severity ratio and delta thresholds to be calibrated more precisely.

The residual false positives in scenarios 3, 5, 6, and 9 deserve visual review. Some may represent genuinely anomalous behavior outside the labeled windows, while others may indicate early detection. These diagnoses have different implications for parameter tuning and should be separated case by case.

The dataset includes the `uptime` column. Uptime-based filters were tested, inspired by Singh et al. (2019), but they did not improve the aggregate metrics in this dataset. A softer approach, such as separate baselines by operating regime or an uptime-aware confidence factor, may be more appropriate than discarding windows directly.

The complexity analysis also suggests one clear engineering improvement for production: optimize sliding-window construction. The current implementation is acceptable for the provided dataset, but a pointer-based or indexed time-series approach would avoid repeated scans over overlapping windows, reducing the cost of that immutable pipeline stage without changing model behavior.

Finally, with more historical data per asset, representation-based methods such as LSTM-AE or RBF-kernel SVM, discussed by Vos et al. (2022) and Radicioni et al. (2025), could capture temporal patterns that fixed z-score thresholds cannot reach. The current approach has the advantage of being fully explainable and not requiring labeled data for training, which is relevant in industrial contexts with few historical incidents.

## References

SINGH, A.; SANKARAN, S.; AMBRE, S.; SRIKONDA, R.; HOUSTON, Z. Improving Deepwater Facility Uptime Using Machine Learning Approach. In: SPE Annual Technical Conference and Exhibition, Calgary, Alberta, Canada, 2019. SPE-195875-MS.

KAMAT, P.; SUGANDHI, R. Anomaly Detection for Predictive Maintenance in Industry 4.0: A Survey. E3S Web of Conferences, v. 170, artigo 02007, 2020. DOI: 10.1051/e3sconf/202017002007.

VOS, K.; PENG, Z.; JENKINS, C.; SHAHRIAR, M. R.; BORGHESANI, P.; WANG, W. Vibration-Based Anomaly Detection Using LSTM/SVM Approaches. Mechanical Systems and Signal Processing, v. 169, artigo 108752, 2022. DOI: 10.1016/j.ymssp.2021.108752.

HU, D.; ZHANG, C.; YANG, T.; CHEN, G. An Intelligent Anomaly Detection Method for Rotating Machinery Based on Vibration Vectors. IEEE Sensors Journal, v. 22, n. 14, p. 14294-14305, 2022. DOI: 10.1109/JSEN.2022.3179740.

RADICIONI, L.; BONO, F. M.; CINQUEMANI, S. Vibration-Based Anomaly Detection in Industrial Machines: A Comparison of Autoencoders and Latent Spaces. Machines, v. 13, artigo 139, 2025. DOI: 10.3390/machines13020139.
