# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=1, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.02122 +/- 0.064 | -1.357e-05 +/- 4.1e-05 | 696.8 +/- 1.5e+02 | 32.04 +/- 8.2 |
| water_absolute+momentum | 5 | -0.001874 +/- 0.054 | -1.206e-06 +/- 3.4e-05 | 735 +/- 1.6e+02 | 31.36 +/- 9.9 |
