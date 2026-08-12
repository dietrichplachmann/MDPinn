# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=5.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 7.645 +/- 19 | 0.004891 +/- 0.012 | 967.7 +/- 86 | 39.1 +/- 4.1 |
| water_absolute+momentum | 5 | 4.044 +/- 6.5 | 0.002587 +/- 0.0042 | 1036 +/- 20 | 45.04 +/- 5.7 |
