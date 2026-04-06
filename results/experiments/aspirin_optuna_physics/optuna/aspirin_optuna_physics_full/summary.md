# aspirin_optuna_physics_full

- Generated: 2026-04-03T05:37:50.935873
- Mode: physics
- Objective metric: val_total_mse_loss
- Objective direction: minimize
- Completed trials: 1
- Pruned trials: 7
- Failed trials: 0

## Best Trial

- Trial: 0
- Objective score: 114.12541198730469
- Objective metric value: 114.12541198730469
- Trial folder: `results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_full/trial_0000`
- Checkpoint: `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_full/trial_0000/checkpoints/trial_0000.ckpt`

### Selected Parameters

- `batch_size` = `32`
- `lr` = `0.000291063591313307`
- `weight_decay` = `2.4810409748678096e-06`
- `embedding_dimension` = `64`
- `num_layers` = `4`
- `num_rbf` = `64`
- `energy_weight` = `0.014940278630992588`
- `force_weight` = `0.9827783645188786`
- `momentum_weight` = `0.04162213204002109`
- `nve_weight` = `0.010616955533913808`
- `baseline_epsilon_eV` = `0.0020366442026830914`
- `baseline_sigma_A` = `0.7751067647801507`
- `baseline_cutoff_A` = `4.521211214797688`
- `traj_length` = `20`
- `nve_freq` = `10`
- `nve_warmup_epochs` = `2`
- `nve_ramp_epochs` = `1`
- `nve_relative` = `False`
- `nve_relative_eps` = `2.692646910086179e-06`
- `nve_loss_mode` = `total_energy`
- `nve_dt_fs` = `0.5`

## Top Trials

| rank | trial | objective | status | metric | checkpoint |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 114.12541198730469 | completed | 114.12541198730469 | `/home/lachmadh/projects/MDPinn/results/experiments/aspirin_optuna_physics/optuna/aspirin_optuna_physics_full/trial_0000/checkpoints/trial_0000.ckpt` |
