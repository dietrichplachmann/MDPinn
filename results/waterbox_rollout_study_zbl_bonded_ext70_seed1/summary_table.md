# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=1, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 0.02434 +/- 0.022 | 1.557e-05 +/- 1.4e-05 | 772.2 +/- 15 | 30.16 +/- 5.3 |
| water_absolute+momentum | 5 | 0.04853 +/- 0.013 | 3.105e-05 +/- 8.1e-06 | 705.9 +/- 18 | 28.07 +/- 2.4 |
