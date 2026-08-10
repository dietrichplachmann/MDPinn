# Water-box study summary (mean +/- std across seeds)

Expected sanity check: water_absolute's mean_per_molecule_momentum_violation
should be clearly nonzero (unlike the aspirin single-molecule study, where the
whole-molecule version of this quantity was ~1e-9, floating-point noise, even
with no training pressure on it). If it's already ~0 here too, that undercuts
this study's premise and is worth knowing before reading anything else below.

n=3 seeds supports a coarse signal-vs-noise read, not formal significance -
treat a difference as real only if it clears roughly 1 std of both conditions.

| condition | n_seeds | energy_mae | force_mae | mean_per_molecule_momentum_violation | max_per_molecule_momentum_violation | final_epoch | total_wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| water_absolute | 3 | 8.919 +/- 6.6 | 1.256 +/- 1.1 | 81.41 +/- 64 | 1859 +/- 1e+03 | 48 +/- 0 | 1.288e+04 +/- 25 |
| water_absolute+momentum | 3 | 6.176 +/- 1.3 | 0.6311 +/- 0.039 | 43.09 +/- 4 | 1526 +/- 3.7e+02 | 48 +/- 0 | 2.603e+04 +/- 60 |
