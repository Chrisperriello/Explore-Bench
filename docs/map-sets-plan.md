# Plan: frozen map sets (step 1 of [research-plan.md](research-plan.md))

Goal: turn the room generator (scripts/generate_room_maps.py) and the
HouseExpo preview (scripts/preview_houseexpo_maps.py) into saved map sets
that experiments can train and test on, with computed difficulty tags.

## How a set is saved

- One directory per set: `maps/<name>_v<N>/`, holding the `.pgm` files and a
  `manifest.json`.
- The manifest records the source (rooms or HouseExpo), generator commit,
  seed and parameters, and per map: file name, SHA-256, split
  (train/validation/test), tags and validation results.
- Sets are write-once. The writer refuses an existing directory; any change
  becomes `_v<N+1>`. Tasks, checkpoints and manifests already store each
  map's SHA-256 (`task_for`, explore_bench/config.py), so an edited map
  could not silently reuse old results anyway.
- Generated room maps are committed: 250 x 250 bytes is about 62 KB per
  map, so 100 maps is about 6 MB, and Ada gets them with `git pull`.
- HouseExpo-derived maps: the dataset licence has not been checked. Until it
  is, commit only the manifest and rebuild the maps on Ada from the pinned
  archive (`ARCHIVE_SHA256`), refusing any map whose hash differs.

## How it is cleaned up

- The writer builds the set in a temporary directory, validates every map,
  and only renames the whole directory into `maps/` if all checks pass. On
  any failure the temporary directory is deleted.
- Rejected candidates are never saved; the manifest records how many were
  tried and why each was rejected.
- The two preview scripts keep their current behaviour (temporary only).
- Pilot or scratch sets go in `results/maps/` (gitignored) and can be
  deleted freely. Only sets an experiment uses are committed to `maps/`.

## How tagging works

Tags are computed from the `.pgm` by `tags()` in explore_bench/map_tags.py,
used for every source and also run on the 12 hand-made maps so old and new
maps share one scale (`python -m explore_bench.map_tags MAP.pgm ...`):

| Tag | Meaning |
|---|---|
| `free_area_m2` | free cells x 0.01 m² |
| `free_components` | separate free regions; should be 1 |
| `loops` | wall islands fully surrounded by free space, any size (square_loop = 1, room_1 = 6) |
| `largest_island_m2` | area of the biggest island: about 1 m² for a wall stub, 200 m² for square_loop's core |
| `narrow_share` | share of free cells less than 0.5 m from a wall |
| `geodesic_diameter_m` | longest 4-connected walk through free space (two-sweep estimate) |
| `free_height_m`, `free_width_m`, `extent` | free-space bounding box; `small` or `full` (for step 2) |

Room and door counts are not computed from the image; the manifest copies
them from the source when it records them (generator, HouseExpo `room_num`).

### Known problem (2026-11-12): tags are labels only, not analysis variables

Until this is fixed, **no layout analysis** (results split or correlated by
tag, room count, rooms vs HouseExpo "structure") may be reported. Tags can
name and sort maps, nothing more. Found from the `show_map_set.py` view of
houseexpo_v1:

- **`rooms` does not describe the drawn map.** For HouseExpo it is the
  source's `room_num`, but rooms exist in the source only as `room_category`
  bounding boxes; the grid is drawn from the `verts` outline alone. Layout
  `9b617c34…` has `room_num` 10, `verts` is a 5-point square, and the map is
  one empty 15 x 16 m room. Measured interior wall per map runs from 0 to
  116 m and only loosely tracks `rooms` (`a5561b…`: 10 rooms, ~24 m of wall).
  Checked against the pinned archive (SHA dad4a7d2…), so this is the source
  data, not the rasterizer. For rooms_v1 courtyard maps `rooms` is also
  approximate.
- **`narrow_share` measures wall perimeter, not corridors.** It is the share
  of free cells within 0.5 m of any wall, so the empty square scores 0.10;
  it mostly grows with wall length per area.
- **`loops` cannot separate HouseExpo layouts.** Every houseexpo_v1 map has 0,
  because interior walls attach to the outline. Topologically correct, but it
  carries no information for that set.

To resolve before any layout figure: derive room/door structure from the
grid itself (e.g. rooms as regions split at doorways), redefine narrowness
as passage width (free-space medial axis), decide whether layouts with
almost no interior wall stay in the test set, and re-check tags by eye
against `show_map_set.py`. Frozen manifests are write-once, so corrected tags
are a recomputed table joined by map hash, not an edit to the sets.

Results such as "Cost steps to 99%" are not tags; they come from baseline
runs. Analysis joins `episodes.csv` (`map_name`) to the manifest, so results
can be reported per tag bin, e.g. success against `loops`.

## How maps are injected

Experiment JSONs can already replace `"maps"` (it is a DEFAULT key in
explore_bench/config.py), and configuration rejects the same map content in
two splits. `prepare` then freezes tasks: 10 start positions per map per
suite (`evaluation_tasks`).

1. **New training runs:** a new `experiments/*.json` listing set maps. No
   code change needed.
2. **Re-testing existing checkpoints on new test maps** (cheapest result, no
   retraining): blocked today. `ActorController` refuses a checkpoint unless
   `digest(spec)` matches its training configuration (explore_bench/mappo.py:58),
   and that includes the map lists. Needs an explicit path that compares
   digests with `maps.test` removed and records the substituted test set in
   the run manifest, so scoring never mixes it with the original test set.

Compute: MMPF runs about 10 steps/min, so 50 test maps x 10 starts = 500
episodes per method and seed. An optional `starts_per_map` key (absent = 10,
so existing digests are unchanged, as with `navigation`) would cut this.

## Steps

1. Tag function, tested on the 12 hand-made maps (square_loop: 1 loop;
   corridor: 0).
2. `scripts/freeze_map_set.py --source rooms|houseexpo --name --count --seed
   --splits` writes a set and its manifest.
3. Check the HouseExpo licence; decide commit or rebuild.
4. Test-set substitution for existing checkpoints.
5. Pilot: freeze a ~30-map rooms test set; on Ada, evaluate Cost and one
   MAPPO seed (360° belief) on it.

## Keeping it fast as maps are added

- Training cost does not grow with the number of maps. Each of the 8
  training worlds (one process per CPU, explore_bench/worlds.py) draws a
  random train map per episode (worlds.py:30), and the run stops at a fixed
  step budget, so 8 or 80 train maps cost the same.
- Evaluation cost grows linearly with maps x starts, but every episode is
  independent. Baselines already run as one Slurm array task per controller
  and map (hpc/jobs/mappo/submit_baselines.sh); sharding per episode instead
  would make wall time depend on free Ada CPUs rather than on map count.
- Averaging the weights of separately trained networks is not used: unlike
  a Q-table, two neural networks from different random starts usually do not
  average into a working network. Parallel experience collection into one
  learner (MAPPO's 8 worlds now; many actors feeding one replay buffer for
  DQN) is the standard speed-up.

## Decisions (2026-11-10)

- Commit generated room `.pgm` files: yes.
- Starts per map: keep 10; recover speed through parallel evaluation.
- First set: about 30 maps.
- Re-testing existing checkpoints on new test maps (step 4): yes.
