# HouseExpo preview

From the repository root:

```bash
.venv/bin/python scripts/preview_houseexpo_maps.py
```

This downloads the pinned HouseExpo archive (about 25 MB) into temporary storage,
samples candidates reproducibly, and opens a window with six passing layouts.
The terminal shows each accepted map's checks and each rejected candidate's
reason. Blue and orange markers indicate the two sampled robot starts.
Candidate `.pgm` files are deleted after validation; downloaded data is deleted
when the window closes. No maps are added to an experiment.

To use an existing archive or extracted JSON directory:

```bash
.venv/bin/python scripts/preview_houseexpo_maps.py --source /path/to/json.tar.gz
.venv/bin/python scripts/preview_houseexpo_maps.py --source /path/to/HouseExpo/json
```

To change the sample or write a contact sheet without opening a window:

```bash
.venv/bin/python scripts/preview_houseexpo_maps.py --seed 42 --count 6
.venv/bin/python scripts/preview_houseexpo_maps.py --preview /tmp/houseexpo-preview.png
```

An explicitly requested preview PNG is kept. Use `--max-attempts 120` if the
default 60 candidates do not yield enough passing layouts. An incomplete
selection displays the available passing maps and exits with status 1. Source,
download, and display I/O errors exit with status 2. Success exits with status 0.

## Conversion and validation

HouseExpo stores a floor contour in `verts`, in meters. The converter follows
the upstream contour-fill convention at 1 cm resolution, then keeps a 10 cm cell
free only when its entire fine-resolution block is free. This retains thin
barriers but can narrow doorways. It centers the result on a 250 × 250 canvas
at 0.1 m/cell, adds a wall boundary, and uses the benchmark's values:
`0` wall, `205` exterior, `254` free space. Oversized layouts are rejected.

Checks cover:

- Exact occupancy format and enclosure of the free region.
- One connected free region using four-neighbor connectivity.
- One surviving region after erosion by a 9 × 9 square footprint (0.9 m).
- Free area of 100–400 m², matching the procedural prototype's selection range.
- Loading through the benchmark's `task_for`, including two reproducible spawns.
- Three real Cost decisions through `Session`, with nondecreasing coverage and
  robot positions remaining in free space. `--smoke-steps 0` skips this check.

The area and erosion checks select a subset of HouseExpo. The erosion check
does not prove that every original free cell has the stated clearance. A short
Cost run checks compatibility; it does not establish full exploration success,
optimal path length, or MAPPO performance. Source room counts are displayed as
metadata, rather than inferred from the rasterized map.

The same source, seed, and options reproduce candidate order, grids, and spawn
positions. Archive members are read directly, without extracting paths to disk.
The default download is pinned to upstream revision `45e2b25` and checked against
SHA-256 `dad4a7d230937bdfe1d57c91330e697b625ec681eefd9cda3159082a939b4f2f`.

Sources: [HouseExpo dataset](https://github.com/TeaganLi/HouseExpo),
[upstream contour reader](https://github.com/TeaganLi/HouseExpo/blob/45e2b25/pseudoslam/envs/simulator/jsonReader.py),
and [HouseExpo paper](https://arxiv.org/abs/1903.09845).
