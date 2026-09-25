# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=2, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.07086 +/- 0.066 | -4.534e-05 +/- 4.2e-05 | 756.4 +/- 20 | 29.87 +/- 1.6 |
| water_absolute+momentum | 5 | -0.02555 +/- 0.035 | -1.634e-05 +/- 2.2e-05 | 846.2 +/- 18 | 34 +/- 3.4 |
