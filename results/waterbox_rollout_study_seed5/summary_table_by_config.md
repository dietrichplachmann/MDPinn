# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=5, use_zbl_prior=False.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.03566 +/- 0.015 | -2.281e-05 +/- 9.8e-06 | 742.4 +/- 1.4e+02 | 27 +/- 2 |
| water_absolute+momentum | 5 | -0.00748 +/- 0.061 | -4.788e-06 +/- 3.9e-05 | 837.6 +/- 1.7e+02 | 40.17 +/- 17 |
