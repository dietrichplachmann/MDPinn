# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=0, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.06476 +/- 0.083 | -4.143e-05 +/- 5.3e-05 | 1682 +/- 89 | 74.46 +/- 11 |
| water_absolute+momentum | 5 | -0.09272 +/- 0.037 | -5.932e-05 +/- 2.3e-05 | 1072 +/- 22 | 40.52 +/- 2.2 |
