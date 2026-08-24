# Water-box study summary (mean +/- std across seeds)

Expected sanity check: water_absolute's mean_per_molecule_momentum_violation
should be clearly nonzero (unlike the aspirin single-molecule study, where the
whole-molecule version of this quantity was ~1e-9, floating-point noise, even
with no training pressure on it). If it's already ~0 here too, that undercuts
this study's premise and is worth knowing before reading anything else below.

n=6 seeds supports a coarse signal-vs-noise read, not formal significance - treat a difference as real only if it clears roughly 1 std of both conditions.

| condition | n_seeds | energy_mae | force_mae | mean_per_molecule_momentum_violation | max_per_molecule_momentum_violation | final_epoch | total_wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| water_absolute | 6 | 2679 +/- 6.6e+03 | 1.06 +/- 0.96 | 64.12 +/- 37 | 943.2 +/- 5e+02 | 48 +/- 0 | 1.297e+04 +/- 51 |
| water_absolute+momentum | 6 | 3.829 +/- 1.2 | 0.6687 +/- 0.1 | 49.1 +/- 8.8 | 1034 +/- 5.3e+02 | 48 +/- 0 | 2.649e+04 +/- 72 |
