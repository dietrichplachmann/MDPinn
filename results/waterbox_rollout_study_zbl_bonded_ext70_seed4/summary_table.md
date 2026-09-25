# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=4, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.03292 +/- 0.085 | -2.106e-05 +/- 5.5e-05 | 783.3 +/- 32 | 32.63 +/- 4.6 |
| water_absolute+momentum | 5 | 0.06476 +/- 0.086 | 4.143e-05 +/- 5.5e-05 | 765.1 +/- 13 | 28.59 +/- 2.8 |
