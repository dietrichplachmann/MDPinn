# Water-box study summary (mean +/- std across seeds)

Expected sanity check: water_absolute's mean_per_molecule_momentum_violation
should be clearly nonzero (unlike the aspirin single-molecule study, where the
whole-molecule version of this quantity was ~1e-9, floating-point noise, even
with no training pressure on it). If it's already ~0 here too, that undercuts
this study's premise and is worth knowing before reading anything else below.

n=6 seeds supports a coarse signal-vs-noise read, not formal significance - treat a difference as real only if it clears roughly 1 std of both conditions.

| condition | n_seeds | energy_mae | force_mae | mean_per_molecule_momentum_violation | max_per_molecule_momentum_violation | final_epoch | total_wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| water_absolute | 6 | 5.941 +/- 5.3 | 0.9637 +/- 0.78 | 60.87 +/- 46 | 1223 +/- 9.7e+02 | 48 +/- 0 | 1.285e+04 +/- 35 |
| water_absolute+momentum | 6 | 4.794 +/- 2 | 0.6354 +/- 0.037 | 43.83 +/- 5.3 | 1394 +/- 8.7e+02 | 48 +/- 0 | 2.603e+04 +/- 47 |
