# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=2, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.07945 +/- 0.038 | -5.082e-05 +/- 2.4e-05 | 635.8 +/- 1.5e+02 | 28.05 +/- 5.7 |
| water_absolute+momentum | 5 | -0.01287 +/- 0.036 | -8.23e-06 +/- 2.3e-05 | 668.1 +/- 1.5e+02 | 28.48 +/- 7.5 |
