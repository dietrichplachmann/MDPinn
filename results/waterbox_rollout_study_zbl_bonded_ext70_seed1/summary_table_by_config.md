# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=1, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.02441 +/- 0.039 | -1.562e-05 +/- 2.5e-05 | 643.7 +/- 1.5e+02 | 29.06 +/- 5.6 |
| water_absolute+momentum | 5 | -0.02591 +/- 0.025 | -1.657e-05 +/- 1.6e-05 | 611.5 +/- 1.3e+02 | 28.63 +/- 7.4 |
