# Ablation summary (mean +/- std across seeds)

n=3 seeds supports a coarse signal-vs-noise read, not formal significance -
treat a difference as real only if it clears roughly 1 std of both conditions.

| molecule | condition | n_seeds | energy_mae | force_mae | rollout_mean_max_abs_drift_eV | rollout_failure_rate | structural_stability_score | wall_seconds_to_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aspirin | absolute | 1 | 1.868 +/- 0 | 3.726 +/- 0 | 18.49 +/- 0 | 0 +/- 0 | 0.4028 +/- 0 | 5161 +/- 0 |
