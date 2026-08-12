# Water-box rollout stability study summary (mean +/- std across velocity-draw replicates)

n=5 velocity-seed replicates per condition, identical starting geometry (DATA_SEED/TEST_CONFIG_INDEX held fixed - only the initial Maxwell-Boltzmann velocity draw differs between replicates). A single matched trial already flipped which condition showed less drift once a velocity-draw confound was controlled for - read a difference here as a coarse signal, not a formal significance claim, the same way the static-metric study treats n=3-6 training seeds.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| water_absolute | 5 | 28.09 +/- 28 | 0.01798 +/- 0.018 | 1949 +/- 84 | 79.31 +/- 9 |
| water_absolute+momentum | 5 | 64.91 +/- 27 | 0.04154 +/- 0.017 | 2317 +/- 1.2e+02 | 103.9 +/- 3 |
