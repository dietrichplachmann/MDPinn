# aspirin_optuna_physics_screen

- Generated: 2026-03-30T19:09:31.145038
- Mode: physics
- Objective metric: val_total_mse_loss
- Objective direction: minimize
- Completed trials: 2
- Pruned trials: 22
- Failed trials: 0

## Best Trial

- Trial: 0
- Objective score: 672.432861328125
- Objective metric value: 672.432861328125
- Trial folder: `results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_screen/trial_0000`
- Checkpoint: `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_screen/trial_0000/checkpoints/trial_0000.ckpt`

### Selected Parameters

- `batch_size` = `32`
- `lr` = `0.000291063591313307`
- `weight_decay` = `2.4810409748678096e-06`
- `embedding_dimension` = `64`
- `num_layers` = `2`
- `num_rbf` = `64`
- `energy_weight` = `0.2097862337921012`
- `force_weight` = `0.8009613865627863`
- `momentum_weight` = `0.009091248360355032`
- `nve_weight` = `0.00917022549267169`
- `baseline_epsilon_eV` = `0.0032877474139911193`
- `baseline_sigma_A` = `1.2871346474483567`
- `baseline_cutoff_A` = `5.159725093210579`
- `traj_length` = `30`
- `nve_freq` = `50`
- `nve_warmup_epochs` = `0`
- `nve_ramp_epochs` = `5`
- `nve_relative` = `False`
- `nve_relative_eps` = `6.245139574743075e-05`
- `nve_loss_mode` = `total_energy`
- `nve_dt_fs` = `0.1`

## Top Trials

| rank | trial | objective | status | metric | checkpoint |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 672.432861328125 | completed | 672.432861328125 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_screen/trial_0000/checkpoints/trial_0000.ckpt` |
| 2 | 3 | 8707.6806640625 | completed | 8707.6806640625 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_screen/trial_0003/checkpoints/trial_0003.ckpt` |
