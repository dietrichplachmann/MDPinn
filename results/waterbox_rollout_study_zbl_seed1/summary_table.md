# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=1, use_zbl_prior=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 2.311e+06 +/- 3.1e+06 | 1479 +/- 2e+03 | 4.293e+06 +/- 7.5e+06 | 4.906e+06 +/- 7.4e+06 |
| water_absolute+momentum | 5 | 4.977e+07 +/- 5.5e+07 | 3.184e+04 +/- 3.5e+04 | 1.004e+08 +/- 1.1e+08 | 8.001e+07 +/- 7.9e+07 |
