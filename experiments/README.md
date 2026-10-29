# Experiment configurations

Each JSON overrides `DEFAULT` in `explore_bench/config.py`; every run stores the
fully resolved configuration, so these short files are not the whole record.

| File | Sensor | Navigation | Purpose |
| --- | --- | --- | --- |
| `mappo_omni_3p5_v1.json` | 360°, 3.5 m | belief | Main MAPPO study (`mappo_omni_3p5_full_01`) |
| `mappo_four_beam_3p5_v1.json` | four-beam, 3.5 m | belief | Four-beam pilot and future full study |
| `mappo_omni_3p5_legacy_v1.json` | 360°, 3.5 m | ground_truth | Legacy routing: MAPPO and controls |
| `mappo_four_beam_3p5_legacy_v1.json` | four-beam, 3.5 m | ground_truth | Legacy routing controls for four-beam |

`navigation` is the map A* may route through: `belief` (default) allows only
sensed free space; `ground_truth` reproduces the original Level-0 planner,
which may cross unseen free space. Goals still come from beliefs in both.

Why the legacy files exist: under belief routing Cost/MMPF plateau near
75-78% coverage, while the Oct 7-8 legacy collection shows them finishing
80-87% of runs. Running MAPPO and the controls under both rules tests whether
the method ranking depends on routing. See `NAVIGATION` in
`explore_bench/config.py` and `Session._navigation_map` in
`explore_bench/session.py`. Scoring refuses to mix the two contracts.

Leaving `navigation` out keeps a configuration's digest identical to before
the option existed, which is why the original files do not mention it.
