# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=1, use_zbl_prior=True, zbl_bonded_exclusion=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.003223 +/- 0.047 | -2.061e-06 +/- 3e-05 | 821.3 +/- 15 | 31.51 +/- 2.6 |
| water_absolute+momentum | 5 | -0.03522 +/- 0.028 | -2.253e-05 +/- 1.8e-05 | 901.7 +/- 20 | 38.09 +/- 5.7 |
