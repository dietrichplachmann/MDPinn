# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting configuration (test_config_index) differs between replicates. Compare against summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-absolute separation is a property of the models or of the one configuration it was run on. train_seed=1, use_zbl_prior=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 2.971e+06 +/- 5.5e+06 | 1901 +/- 3.5e+03 | 3.788e+06 +/- 7.4e+06 | 6.94e+06 +/- 1.3e+07 |
| water_absolute+momentum | 5 | 4381 +/- 5e+03 | 2.802 +/- 3.2 | 8473 +/- 6.2e+03 | 5586 +/- 5.5e+03 |
