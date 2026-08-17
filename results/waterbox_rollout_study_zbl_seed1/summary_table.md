# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the initial Maxwell-Boltzmann velocity draw differs between replicates. train_seed=1, use_zbl_prior=True.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | -0.08026 +/- 0.03 | -5.135e-05 +/- 1.9e-05 | 4615 +/- 76 | 171.4 +/- 17 |
| water_absolute+momentum | 5 | -0.01896 +/- 0.061 | -1.213e-05 +/- 3.9e-05 | 1759 +/- 2.1e+02 | 119.8 +/- 31 |
