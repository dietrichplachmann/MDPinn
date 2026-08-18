# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=5, use_zbl_prior=False.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.02133 +/- 0.044 | -1.365e-05 +/- 2.8e-05 | 933.2 +/- 17 | 39.3 +/- 3.8 |
| water_absolute+momentum | 5 | -0.04442 +/- 0.034 | -2.842e-05 +/- 2.2e-05 | 1061 +/- 14 | 53.1 +/- 10 |
