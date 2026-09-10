# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Same training seed/condition, before (before_finetune=checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt) vs. after (after_finetune=checkpoints/waterbox_study_zbl_bonded_ext70_stable_local_v1/water_absolute/seed1/stable_final.ckpt) StABlE fine-tuning. velocity-axis replicates at dt=0.1, matching results/waterbox_rollout_study_zbl_bonded_ext70_seed1's own DATA_SEED/test_config_index exactly for direct comparability.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| after_finetune | 5 | -0.06129 +/- 0.046 | -3.92e-05 +/- 3e-05 | 771.5 +/- 14 | 32.13 +/- 2.1 |
| before_finetune | 5 | -0.03378 +/- 0.044 | -2.161e-05 +/- 2.8e-05 | 768 +/- 26 | 29.97 +/- 1.7 |
