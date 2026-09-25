# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=5, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.06372 +/- 0.057 | -4.076e-05 +/- 3.7e-05 | 763.9 +/- 1.5e+02 | 29.91 +/- 6.8 |
| water_absolute+momentum | 5 | -0.0164 +/- 0.071 | -1.05e-05 +/- 4.6e-05 | 828.8 +/- 1.6e+02 | 30.96 +/- 5.5 |
