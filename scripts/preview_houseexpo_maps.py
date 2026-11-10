"""Preview and validate temporary HouseExpo maps without changing an experiment.

Run: .venv/bin/python scripts/preview_houseexpo_maps.py
Use --source PATH for an existing json.tar.gz or directory of HouseExpo JSONs.
Otherwise the pinned public archive (~25 MB) is downloaded into temporary storage.
--preview PNG saves a contact sheet instead of opening an interactive window.

Selection matches the room-map prototype: 250x250 at 0.1 m/cell, 100-400 m²
free area, one free component, and one component after 9x9-square erosion.
These are subset filters, not a proof that a robot can explore the whole map.
No checkpoint is needed: a short Cost run checks simulator compatibility only.
All downloaded data and .pgm files are deleted when the preview closes.
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARCHIVE_URL = ("https://raw.githubusercontent.com/TeaganLi/HouseExpo/"
               "45e2b25/HouseExpo/json.tar.gz")
ARCHIVE_SHA256 = "dad4a7d230937bdfe1d57c91330e697b625ec681eefd9cda3159082a939b4f2f"
SIZE, RESOLUTION = 250, 0.1
WALL, OUTSIDE, FREE = 0, 205, 254
SUPERSAMPLE = 10  # Draw at 1 cm, then retain only wholly free 10 cm cells.
PADDING = 2
AREA_M2 = (100, 400)
CLEARANCE_CELLS = 9


def download_archive(directory):
    path = directory / "houseexpo_json.tar.gz"
    print("Downloading HouseExpo (~25 MB) into temporary storage...", flush=True)
    with urlopen(ARCHIVE_URL, timeout=60) as response, path.open("wb") as target:
        shutil.copyfileobj(response, target)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprint != ARCHIVE_SHA256:
        raise ValueError("HouseExpo archive checksum differs from the pinned source")
    print("Verified archive SHA-256:", fingerprint, flush=True)
    return path


def candidates(source, seed, limit):
    """Sample sorted names; read tar members directly without extracting files."""
    rng = random.Random(seed)
    if source.is_dir():
        files = sorted(source.rglob("*.json"))
        for path in rng.sample(files, min(limit, len(files))):
            yield path.stem, path.read_bytes()
    else:
        with tarfile.open(str(source), "r:gz") as archive:
            members = sorted((member for member in archive.getmembers()
                              if member.isfile() and member.name.endswith(".json")),
                             key=lambda member: member.name)
            chosen = rng.sample(members, min(limit, len(members)))
            # Reading in archive order avoids repeatedly decompressing gzip.
            payloads = {}
            for member in sorted(chosen, key=lambda member: member.offset_data):
                with archive.extractfile(member) as handle:
                    payloads[member.name] = handle.read()
            for member in chosen:
                yield Path(member.name).stem, payloads[member.name]


def rasterize(house):
    """Preserve scale and thin barriers; reject oversized geometry before drawing.

    HouseExpo's reader fills the contour in `verts`. Use the same contour-fill
    convention at 1 cm resolution. Conservative downsampling can narrow a door
    by a cell; validation reports disconnections instead of repairing the layout.
    A wall ring separates free space from exterior (205). Enclosed solid regions
    are walls, matching the upstream convention that all non-floor is obstacle.
    """
    if not isinstance(house, dict):
        raise ValueError("house must be a JSON object")
    verts = np.asarray(house.get("verts"), dtype=np.float64)
    if (verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3
            or not np.isfinite(verts).all()):
        raise ValueError("verts must contain at least three finite (x, y) points")
    span = np.ptp(verts, axis=0)
    if (span <= 0).any() or (span > (SIZE - 2 * PADDING - 2) * RESOLUTION).any():
        raise ValueError("layout extent {:.2f} x {:.2f} m does not fit at 0.1 m/cell"
                         .format(*span))
    # Translate before quantizing, so a large world-coordinate offset is harmless.
    fine = np.rint((verts - verts.min(axis=0)) / RESOLUTION * SUPERSAMPLE).astype(np.int32)
    dims = (np.ceil((fine.max(axis=0) + 1) / SUPERSAMPLE).astype(int) + 2)
    if (dims > SIZE - 2 * PADDING).any():
        raise ValueError("layout plus boundary does not fit at 0.1 m/cell")
    fine += SUPERSAMPLE  # One empty coarse cell around the contour.
    width, height = map(int, dims)
    mask = np.zeros((height * SUPERSAMPLE, width * SUPERSAMPLE), dtype=np.uint8)
    cv2.drawContours(mask, [fine], 0, 1, -1)
    local_free = mask.reshape(height, SUPERSAMPLE, width, SUPERSAMPLE).all(axis=(1, 3))
    free = np.zeros((SIZE, SIZE), dtype=np.uint8)
    top, left = (SIZE - height) // 2, (SIZE - width) // 2
    free[top:top + height, left:left + width] = local_free
    beyond_wall = (cv2.dilate(free, np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
    cv2.floodFill(beyond_wall, None, (0, 0), 2, flags=4)
    grid = np.full((SIZE, SIZE), WALL, dtype=np.uint8)
    grid[beyond_wall == 2] = OUTSIDE
    grid[free != 0] = FREE
    return grid


def component_count(mask):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[0] - 1


def validate(path, seed, smoke_steps):
    """Check geometry, the real task loader, and optional Cost transitions."""
    from explore_bench.config import configuration, task_for

    grid = np.asarray(Image.open(path))
    valid_format = (grid.shape == (SIZE, SIZE) and grid.dtype == np.uint8
                    and set(np.unique(grid)) <= {WALL, OUTSIDE, FREE})
    checks = [("format", valid_format, "250x250 uint8; values 0/205/254")]
    if not valid_format:
        return checks, []
    free = grid == FREE
    outside = np.pad(grid == OUTSIDE, 1, constant_values=True)
    leaks = free & (outside[:-2, 1:-1] | outside[2:, 1:-1]
                    | outside[1:-1, :-2] | outside[1:-1, 2:])
    checks.append(("enclosed", not leaks.any(), "{} free cells touch outside".format(leaks.sum())))
    count = component_count(free)
    checks.append(("connected", count == 1, "{} free region(s)".format(count)))
    eroded = cv2.erode(free.astype(np.uint8), np.ones((CLEARANCE_CELLS, CLEARANCE_CELLS), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    wide_count = component_count(eroded)
    checks.append(("clearance filter", wide_count == 1,
                   "{} region(s) after 9x9-square erosion (0.9 m footprint)".format(wide_count)))
    area = np.count_nonzero(free) * RESOLUTION ** 2
    checks.append(("area filter", AREA_M2[0] <= area <= AREA_M2[1],
                   "{:.1f} m²; selected range {}-{} m²".format(area, *AREA_M2)))
    # Do not spend planner time on geometry that already fails selection.
    if not all(passed for _, passed, _ in checks):
        return checks, []
    spawns = []
    try:
        spec = configuration()
        if spec["resolution"] != RESOLUTION:
            raise ValueError("benchmark resolution differs from the converter")
        task = task_for(spec, str(path), seed, "houseexpo_preview")
        spawns = task.start_positions
        checks.append(("benchmark loader", True, "two robot starts {}".format(spawns)))
        if smoke_steps:
            from explore_bench.controllers import ClassicalController
            from explore_bench.evaluation import baseline_view
            from explore_bench.session import Session

            session = Session(spec)
            controller = ClassicalController("cost", None)
            controller.bind_view = lambda: baseline_view(session)
            actor, _, available = session.reset_task(task)
            executed = 0
            coverage = session.measurement()["team_coverage_ratio"]
            for _ in range(smoke_steps):
                if session.terminated or session.truncated:
                    break
                request, _ = controller.act(actor, None, available)
                (actor, _, available), _, _, _, measured = session.step_request(request)
                executed += 1
                next_coverage = measured["team_coverage_ratio"]
                if not coverage <= next_coverage <= 1:
                    raise ValueError("invalid coverage during Cost smoke run")
                coverage = next_coverage
                if any(grid[row, column] != FREE for row, column in session.agent_pos):
                    raise ValueError("robot left free space during Cost smoke run")
            checks.append(("Cost smoke run", True,
                           "{} steps; {:.1%} coverage (compatibility only)".format(executed, coverage)))
    except Exception as error:
        checks.append(("benchmark compatibility", False, "{}: {}".format(type(error).__name__, error)))
    return checks, spawns


def show(results, attempted, rejected, preview):
    import matplotlib
    if preview:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    columns = min(3, len(results))
    rows = (len(results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5.4 * rows),
                               squeeze=False, constrained_layout=True)
    colors = ListedColormap(["#26313d", "#e1e0d9", "#ffffff"])
    for ax in axes.flat:
        ax.axis("off")
    for ax, result in zip(axes.flat, results):
        grid = result["grid"]
        ax.imshow(np.select([grid == WALL, grid == FREE], [0, 2], 1), cmap=colors,
                  vmin=0, vmax=2, interpolation="nearest")
        area = np.count_nonzero(grid == FREE) * RESOLUTION ** 2
        ax.set_title("{}  PASS ({} checks)\n{} source rooms | {:.1f} m²"
                     .format(result["id"][:12], len(result["checks"]), result["rooms"], area),
                     loc="left", fontsize=10)
        for index, (row, column) in enumerate(result["spawns"]):
            ax.plot(column, row, "o", color=("#1764c0", "#df5f13")[index], markersize=7)
            ax.annotate("R{}".format(index + 1), (column, row), xytext=(5, 5),
                        textcoords="offset points", fontsize=9)
    figure.suptitle("HouseExpo: {} accepted / {} checked; {} rejected (reasons in terminal)\n"
                    "25 x 25 m canvas; 0.1 m/cell. Temporary PGM files deleted when preview closes."
                    .format(len(results), attempted, rejected), x=0.01, ha="left", fontsize=11)
    try:
        if preview:
            figure.savefig(str(preview), dpi=120)
            print("Saved preview:", preview, flush=True)
        else:
            plt.show()
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="JSON directory or json.tar.gz; default: temporary download")
    parser.add_argument("--count", type=int, default=6, help="number of passing layouts to show (default: 6)")
    parser.add_argument("--seed", type=int, default=1, help="reproducible layout selection and spawns")
    parser.add_argument("--max-attempts", type=int, default=60, help="candidate limit (default: 60)")
    parser.add_argument("--smoke-steps", type=int, default=3, help="Cost steps per passing map; 0 skips")
    parser.add_argument("--preview", type=Path, help="save a PNG instead of opening a window; PNG is kept")
    args = parser.parse_args(argv)
    if args.count < 1 or args.max_attempts < args.count:
        parser.error("count must be positive and max-attempts must be at least count")
    if not 0 <= args.seed < 2 ** 32 or args.smoke_steps < 0:
        parser.error("seed must be a nonnegative 32-bit integer and smoke-steps must be nonnegative")
    if args.source is not None and not args.source.exists():
        parser.error("source does not exist: {}".format(args.source))
    results, attempted, rejected = [], 0, 0
    directory = None
    try:
        with tempfile.TemporaryDirectory(prefix="explore_bench_houseexpo_") as name:
            directory = Path(name)
            source = args.source if args.source is not None else download_archive(directory)
            print("Selection filters: 100-400 m²; 0.9 m square clearance; fixed 0.1 m/cell.", flush=True)
            for source_id, payload in candidates(source, args.seed, args.max_attempts):
                attempted += 1
                path = directory / "candidate.pgm"
                try:
                    house = json.loads(payload)
                    grid = rasterize(house)
                    Image.fromarray(grid).save(str(path))
                    # Tie spawns to the layout ID, independent of the rejection count.
                    seed = int(hashlib.sha256((source_id + ":" + str(args.seed)).encode()).hexdigest()[:8], 16)
                    checks, spawns = validate(path, seed, args.smoke_steps)
                    failures = ["{}: {}".format(check, detail) for check, passed, detail in checks if not passed]
                    if failures:
                        raise ValueError("; ".join(failures))
                    results.append({"id": source_id, "rooms": house.get("room_num", "?"),
                                    "seed": seed, "grid": grid, "checks": checks, "spawns": spawns})
                    print("PASS {} ({} / {}; spawn seed {})"
                          .format(source_id, len(results), args.count, seed), flush=True)
                    for check, _, detail in checks:
                        print("  ok {:<22} {}".format(check, detail), flush=True)
                except (ValueError, TypeError, KeyError) as error:
                    rejected += 1
                    print("REJECT {}: {}".format(source_id, error), flush=True)
                finally:
                    if path.exists():
                        path.unlink()
                if len(results) == args.count:
                    break
            print("Accepted {}, rejected {}, checked {}.".format(len(results), rejected, attempted), flush=True)
            if len(results) < args.count:
                print("Requested {} maps; increase --max-attempts to inspect more candidates."
                      .format(args.count), flush=True)
            if results:
                show(results, attempted, rejected, args.preview)
    except (OSError, ValueError, tarfile.TarError) as error:
        print("Preview failed: {}. Use --source PATH for an offline archive or JSON directory."
              .format(error), file=sys.stderr)
        return 2
    finally:
        if directory is not None:
            print("Deleted temporary storage:", directory, flush=True)
    return 0 if len(results) == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
