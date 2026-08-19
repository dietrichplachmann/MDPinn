# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=5, use_zbl_prior=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.2453 +/- 0.14 | -0.0001569 +/- 8.8e-05 | 5066 +/- 1.6e+02 | 200 +/- 10 |
| water_absolute+momentum | 5 | -0.1994 +/- 0.12 | -0.0001275 +/- 7.8e-05 | 5375 +/- 1.8e+02 | 225.4 +/- 8 |
