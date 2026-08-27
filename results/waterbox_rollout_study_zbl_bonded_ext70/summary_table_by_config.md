# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=0, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.03754 +/- 0.04 | -2.401e-05 +/- 2.6e-05 | 1122 +/- 2.6e+02 | 46.39 +/- 15 |
| water_absolute+momentum | 5 | -0.02197 +/- 0.059 | -1.405e-05 +/- 3.8e-05 | 828.5 +/- 1.7e+02 | 32.25 +/- 8.2 |
