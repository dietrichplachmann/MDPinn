# Water-box study summary (mean +/- std across seeds)

Expected sanity check: water_absolute's mean_per_molecule_momentum_violation
should be clearly nonzero (unlike the aspirin single-molecule study, where the
whole-molecule version of this quantity was ~1e-9, floating-point noise, even
with no training pressure on it). If it's already ~0 here too, that undercuts
this study's premise and is worth knowing before reading anything else below.

n=6 seeds supports a coarse signal-vs-noise read, not formal significance - treat a difference as real only if it clears roughly 1 std of both conditions.

| condition | n_seeds | energy_mae | force_mae | mean_per_molecule_momentum_violation | max_per_molecule_momentum_violation | final_epoch | total_wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| water_absolute | 6 | 3.849 +/- 1.9 | 0.7658 +/- 0.097 | 56.27 +/- 14 | 1358 +/- 7.5e+02 | 67.33 +/- 1.6 | 1.782e+04 +/- 3.9e+02 |
| water_absolute+momentum | 6 | 3.158 +/- 1.6 | 0.73 +/- 0.092 | 49.79 +/- 11 | 1041 +/- 8.1e+02 | 68 +/- 0 | 3.653e+04 +/- 4e+02 |
