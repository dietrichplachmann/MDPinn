# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=1, use_zbl_prior=False.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.0323 +/- 0.021 | -2.066e-05 +/- 1.4e-05 | 843.3 +/- 1.4e+02 | 34.66 +/- 9.1 |
| water_absolute+momentum | 5 | -0.003665 +/- 0.041 | -2.347e-06 +/- 2.7e-05 | 790.1 +/- 1.5e+02 | 34.77 +/- 3.5 |
