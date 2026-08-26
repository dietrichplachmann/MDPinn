# Water-box study summary (mean +/- std across seeds)

Expected sanity check: water_absolute's mean_per_molecule_momentum_violation
should be clearly nonzero (unlike the aspirin single-molecule study, where the
whole-molecule version of this quantity was ~1e-9, floating-point noise, even
with no training pressure on it). If it's already ~0 here too, that undercuts
this study's premise and is worth knowing before reading anything else below.

n=2 seeds supports a coarse signal-vs-noise read, not formal significance - treat a difference as real only if it clears roughly 1 std of both conditions.

| condition | n_seeds | energy_mae | force_mae | mean_per_molecule_momentum_violation | max_per_molecule_momentum_violation | final_epoch | total_wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| water_absolute | 2 | 3.004 +/- 0.78 | 0.7682 +/- 0.042 | 61.39 +/- 1.8 | 2221 +/- 1.9e+02 | 66 +/- 2.8 | 1.757e+04 +/- 7.6e+02 |
| water_absolute+momentum | 2 | 2.375 +/- 0.23 | 0.6893 +/- 0.022 | 54.95 +/- 9.5 | 1836 +/- 1e+03 | 68 +/- 0 | 3.698e+04 +/- 2.2e+02 |
