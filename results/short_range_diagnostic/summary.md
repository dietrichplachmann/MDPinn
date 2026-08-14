# Short-range collapse diagnostic (Q4, paper/main.tex sec:q4)

Empirical non-bonded distance floors (p1.0 over reference DFT configs):
- O-O: floor = 1.7693 A (true min of per-config minima = 1.5175 A, n_reference_configs=160)
- O-H: floor = 0.9743 A (true min of per-config minima = 0.9529 A, n_reference_configs=160)
- H-H: floor = 0.9077 A (true min of per-config minima = 0.7934 A, n_reference_configs=160)

n=42 trajectories analyzed. 15 precede heating onset, 11 coincide, 16 follow, 0 show no sub-floor violation at all.

| label | condition | onset (fs) | first sub-floor (fs) | verdict |
| --- | --- | --- | --- | --- |
| waterbox_rollout | absolute | 55.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_momentum | momentum | 60.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg1 | absolute | 95.0 | 110.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg2 | absolute | 170.0 | 85.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg3 | absolute | 25.0 | 135.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg4 | absolute | 35.0 | 70.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg5 | absolute | 65.0 | 170.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed0 | absolute | 45.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed1 | absolute | 80.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed2 | absolute | 45.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed3 | absolute | 70.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed4 | absolute | 65.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg1 | momentum | 110.0 | 90.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg2 | momentum | 200.0 | 80.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg3 | momentum | 65.0 | 115.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg4 | momentum | 30.0 | 65.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg5 | momentum | 70.0 | 160.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed0 | momentum | 95.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed1 | momentum | 115.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed2 | momentum | 70.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed3 | momentum | 75.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed4 | momentum | 90.0 | 5.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg1 | absolute | 5.0 | 265.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg2 | absolute | 10.0 | 280.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg3 | absolute | 5.0 | 295.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg4 | absolute | 5.0 | 105.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg5 | absolute | 5.0 | 285.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed0 | absolute | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed1 | absolute | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed2 | absolute | 5.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed3 | absolute | 5.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed4 | absolute | 5.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg1 | momentum | 5.0 | 90.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg2 | momentum | 10.0 | 125.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg3 | momentum | 10.0 | 65.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg4 | momentum | 15.0 | 90.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg5 | momentum | 15.0 | 190.0 | short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed0 | momentum | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed1 | momentum | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed2 | momentum | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed3 | momentum | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed4 | momentum | 10.0 | 5.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
