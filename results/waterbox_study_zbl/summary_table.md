# Water-box study summary (mean +/- std across seeds)

Expected sanity check: water_absolute's mean_per_molecule_momentum_violation
should be clearly nonzero (unlike the aspirin single-molecule study, where the
whole-molecule version of this quantity was ~1e-9, floating-point noise, even
with no training pressure on it). If it's already ~0 here too, that undercuts
this study's premise and is worth knowing before reading anything else below.

n=6 seeds supports a coarse signal-vs-noise read, not formal significance - treat a difference as real only if it clears roughly 1 std of both conditions.

| condition | n_seeds | energy_mae | force_mae | mean_per_molecule_momentum_violation | max_per_molecule_momentum_violation | final_epoch | total_wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| water_absolute | 6 | 2737 +/- 6.7e+03 | 1.867 +/- 2.5 | 132.1 +/- 1.7e+02 | 2257 +/- 2.1e+03 | 48 +/- 0 | 1.287e+04 +/- 14 |
| water_absolute+momentum | 6 | 3.795 +/- 1.1 | 0.8198 +/- 0.13 | 51.42 +/- 6.5 | 1329 +/- 5.4e+02 | 48 +/- 0 | 2.633e+04 +/- 28 |
