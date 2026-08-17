# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=1, use_zbl_prior=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.1373 +/- 0.029 | -8.78e-05 +/- 1.9e-05 | 4598 +/- 1.9e+02 | 170.1 +/- 29 |
| water_absolute+momentum | 5 | -0.05427 +/- 0.082 | -3.472e-05 +/- 5.2e-05 | 2511 +/- 6.5e+02 | 266.1 +/- 53 |
