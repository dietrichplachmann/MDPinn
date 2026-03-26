# aspirin_optuna_physics

- Generated: 2026-03-25T19:17:31.698301
- Mode: physics
- Objective metric: val_total_mse_loss
- Objective direction: minimize
- Completed trials: 4
- Pruned trials: 6
- Failed trials: 0

## Best Trial

- Trial: 9
- Objective score: 77.34851837158203
- Objective metric value: 77.34851837158203
- Trial folder: `results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics/trial_0009`
- Checkpoint: `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics/trial_0009/checkpoints/trial_0009.ckpt`

### Selected Parameters

- `batch_size` = `32`
- `lr` = `6.266275266989062e-05`
- `weight_decay` = `2.9067012153229084e-06`
- `embedding_dimension` = `64`
- `num_layers` = `6`
- `num_rbf` = `32`
- `energy_weight` = `0.026936499376103165`
- `force_weight` = `0.9041806267695157`
- `momentum_weight` = `0.0013255655270810907`
- `nve_weight` = `0.029288779063673165`
- `baseline_epsilon_eV` = `0.03957518715404545`
- `baseline_sigma_A` = `1.3632112668138183`
- `baseline_cutoff_A` = `4.940849631032609`
- `traj_length` = `20`
- `nve_freq` = `50`
- `nve_warmup_epochs` = `0`
- `nve_ramp_epochs` = `1`
- `nve_relative` = `False`
- `nve_relative_eps` = `1.8875676307822242e-07`
- `nve_loss_mode` = `total_energy`
- `nve_dt_fs` = `0.1`

## Top Trials

| rank | trial | objective | status | metric | checkpoint |
| --- | --- | --- | --- | --- | --- |
| 1 | 9 | 77.34851837158203 | completed | 77.34851837158203 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics/trial_0009/checkpoints/trial_0009.ckpt` |
| 2 | 1 | 86.156494140625 | completed | 86.156494140625 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics/trial_0001/checkpoints/trial_0001.ckpt` |
| 3 | 2 | 660.0873413085938 | completed | 660.0873413085938 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics/trial_0002/checkpoints/trial_0002.ckpt` |
| 4 | 0 | 9209.6181640625 | completed | 9209.6181640625 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics/trial_0000/checkpoints/trial_0000.ckpt` |
