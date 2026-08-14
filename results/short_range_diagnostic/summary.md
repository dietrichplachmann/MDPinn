# Short-range collapse diagnostic (Q4, paper/main.tex sec:q4)

Empirical non-bonded distance floors (p0.1 over reference DFT configs):
- O-O: floor = 2.5245 A (true min observed = 1.5175 A, n=645120)
- O-H: floor = 1.6336 A (true min observed = 0.9529 A, n=1290602)
- H-H: floor = 1.4399 A (true min observed = 0.7934 A, n=2600960)

n=42 trajectories analyzed. 24 precede heating onset, 18 coincide, 0 follow, 0 show no sub-floor violation at all.

| label | condition | onset (fs) | first sub-floor (fs) | verdict |
| --- | --- | --- | --- | --- |
| waterbox_rollout | absolute | 55.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_momentum | momentum | 60.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg1 | absolute | 95.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg2 | absolute | 170.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg3 | absolute | 25.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg4 | absolute | 35.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/cfg5 | absolute | 65.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed0 | absolute | 45.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed1 | absolute | 80.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed2 | absolute | 45.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed3 | absolute | 70.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute/vseed4 | absolute | 65.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg1 | momentum | 110.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg2 | momentum | 200.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg3 | momentum | 65.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg4 | momentum | 30.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/cfg5 | momentum | 70.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed0 | momentum | 95.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed1 | momentum | 115.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed2 | momentum | 70.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed3 | momentum | 75.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study/runs/water_absolute+momentum/vseed4 | momentum | 90.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg1 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg2 | absolute | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg3 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg4 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/cfg5 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed0 | absolute | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed1 | absolute | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed2 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed3 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute/vseed4 | absolute | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg1 | momentum | 5.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg2 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg3 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg4 | momentum | 15.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/cfg5 | momentum | 15.0 | 0.0 | short-range collapse PRECEDES heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed0 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed1 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed2 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed3 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
| waterbox_rollout_study_seed5/runs/water_absolute+momentum/vseed4 | momentum | 10.0 | 0.0 | short-range collapse COINCIDES with heating onset - consistent with being the trigger |
