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
| water_absolute | 1 | 1.606e+04 +/- 0 | 2.964 +/- 0 | 126.2 +/- 0 | 1262 +/- 0 | 0 +/- 0 | 514.5 +/- 0 |
