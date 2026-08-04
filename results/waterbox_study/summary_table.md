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
| water_absolute | 3 | 8.919 +/- 6.6 | 1.256 +/- 1.1 | 81.41 +/- 64 | 1859 +/- 1e+03 | 18 +/- 0 | 5132 +/- 8.1 |
