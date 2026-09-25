# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=5, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.002389 +/- 0.017 | -1.528e-06 +/- 1.1e-05 | 847.7 +/- 15 | 33.69 +/- 4.9 |
| water_absolute+momentum | 5 | -0.01883 +/- 0.035 | -1.205e-05 +/- 2.2e-05 | 994.9 +/- 18 | 39.05 +/- 2.7 |
