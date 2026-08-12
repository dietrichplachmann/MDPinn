# Force-decomposition replication study: momentum-minus-absolute net-force gap by frame

gap = mean per-molecule net force on water_absolute+momentum's own trajectory geometry minus the same on water_absolute's own trajectory geometry, at matched elapsed time (averaged over which model did the evaluating - the two models agree closely point-wise, see analyze_force_decomposition.py's equilibrium/trajectory results). n_positive = how many replicates (out of n_replicates, pooled across the velocity and config axes) showed momentum's geometry with the higher net force at that frame - the direct replication check for the pattern first noticed in a single trajectory pair.

| frame | n_replicates | gap_mean | gap_std | n_positive |
| --- | --- | --- | --- | --- |
| 0 | 10 | 4.47e-08 | 1.9e-07 | 4/10 |
| 20 | 10 | 0.05024 | 0.153 | 6/10 |
| 80 | 10 | 0.08462 | 0.163 | 8/10 |
| 180 | 10 | 0.3507 | 0.261 | 10/10 |
