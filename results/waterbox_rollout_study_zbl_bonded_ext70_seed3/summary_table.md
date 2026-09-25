# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=3, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.02105 +/- 0.042 | -1.346e-05 +/- 2.7e-05 | 858.3 +/- 21 | 34.43 +/- 5.3 |
| water_absolute+momentum | 5 | 0.3828 +/- 0.85 | 0.0002449 +/- 0.00055 | 820.8 +/- 32 | 31.54 +/- 2.7 |
