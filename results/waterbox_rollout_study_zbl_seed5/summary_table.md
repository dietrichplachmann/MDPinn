# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=5, use_zbl_prior=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.1312 +/- 0.096 | -8.395e-05 +/- 6.1e-05 | 5052 +/- 1.1e+02 | 194.7 +/- 11 |
| water_absolute+momentum | 5 | -0.1409 +/- 0.063 | -9.012e-05 +/- 4e-05 | 5285 +/- 81 | 207.7 +/- 17 |
