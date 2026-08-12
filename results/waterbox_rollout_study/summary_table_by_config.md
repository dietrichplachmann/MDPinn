# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 3.978 +/- 2.7 | 0.002545 +/- 0.0017 | 1285 +/- 1.8e+02 | 56.94 +/- 9.7 |
| water_absolute+momentum | 5 | 7.561 +/- 2.5 | 0.004836 +/- 0.0016 | 1459 +/- 1.4e+02 | 61.59 +/- 13 |
