# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=5.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.7362 +/- 0.35 | -0.0004709 +/- 0.00022 | 738.6 +/- 1.3e+02 | 32.47 +/- 3.7 |
| water_absolute+momentum | 5 | 0.6971 +/- 3.4 | 0.0004458 +/- 0.0021 | 895.6 +/- 2e+02 | 45.66 +/- 11 |
