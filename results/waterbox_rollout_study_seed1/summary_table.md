# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=1, use_zbl_prior=False.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 0.521 +/- 1.2 | 0.0003333 +/- 0.00077 | 1342 +/- 16 | 54.91 +/- 3.2 |
| water_absolute+momentum | 5 | 0.1689 +/- 0.56 | 0.0001081 +/- 0.00036 | 1227 +/- 39 | 53.78 +/- 4.8 |
