"""Generate room-and-door maps, validate them with the benchmark's own code, show them, then delete them.

Usage: python scripts/generate_room_maps.py [--count 6] [--seed 1] [--extra-doors 0]
                                            [--courtyard] [--smoke-steps 3] [--preview PNG]

Why: the benchmark has only 2 unseen test maps (20 episodes per planner), and
both are door-free loops, so "unseen" tests one new map shape rather than
generalization. Generated maps could supply many test maps of controlled
difficulty. This first version only proves the generator produces maps the
pipeline accepts: the .pgm files live in a temporary directory that is deleted
when the window closes, so nothing is added to the benchmark.

Maps copy the format of the hand-made ones (room_1.pgm measured 2026-11-06):
250 x 250 cells at 0.1 m, 205 = outside, 254 = free, 0 = wall, a 1-cell outer
wall, 2-cell inner walls and doors of about 1.4 m. Rooms come from recursive
splitting (binary space partition): each split adds one wall with one door, so
the rooms always form a tree. --extra-doors cuts more doors to create loops.

--courtyard puts a solid block in the middle of the building and splits only
the ring of space around it, so every map has one large loop like the core of
square_loop (166-202 m² islands in the test maps). Extra doors alone only
loop around short wall pieces of a few m².
"""

import argparse
import random
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

# Run from a checkout: validation imports the benchmark's own loaders.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZE, OUTSIDE, WALL, FREE = 250, 205, 0, 254
MARGIN = 20             # gray border around the building, in cells
OUTER_WALL, INNER_WALL = 1, 2
MIN_ROOM = 30           # narrowest room side, 3 m
DOOR = (10, 16)         # door width range in cells, 1.0-1.6 m
DOOR_END_GAP = 4        # keep doors this far from the wall's ends
DOOR_PAD = 3            # new walls stay this far from existing doors
MIN_PASSAGE = 9         # validation: no passage narrower than 0.9 m
AREA_M2 = (100, 400)    # validation: the hand-made maps span 118-380 m² free
MIN_COURTYARD = 40      # smallest courtyard side, 4 m


def generate(rng, extra_doors=0, max_depth=4, courtyard=False):
    grid = np.full((SIZE, SIZE), OUTSIDE, dtype=np.uint8)
    height, width = rng.randint(140, 195), rng.randint(140, 195)
    top, left = rng.randint(MARGIN, SIZE - MARGIN - height), rng.randint(MARGIN, SIZE - MARGIN - width)
    grid[top:top + height, left:left + width] = WALL
    grid[top + OUTER_WALL:top + height - OUTER_WALL, left + OUTER_WALL:left + width - OUTER_WALL] = FREE
    doors, walls = [], []

    def touches_door(r0, c0, r1, c1):
        return any(r0 < d1 and d0 < r1 and c0 < e1 and e0 < c1 for d0, e0, d1, e1 in doors)

    def split(r0, c0, r1, c1, depth):
        rows, columns = r1 - r0, c1 - c0
        if depth >= max_depth or max(rows, columns) < 2 * MIN_ROOM + INNER_WALL:
            return
        if depth >= 2 and rng.random() < 0.3:
            return
        # Split across the longer side so rooms stay roughly square.
        vertical = columns > rows if abs(columns - rows) > 10 else rng.random() < 0.5
        along = columns if vertical else rows
        if along < 2 * MIN_ROOM + INNER_WALL:
            vertical, along = not vertical, rows if vertical else columns
        for _ in range(20):
            cut = rng.randint(MIN_ROOM, along - MIN_ROOM - INNER_WALL)
            wall = ((r0, c0 + cut, r1, c0 + cut + INNER_WALL) if vertical
                    else (r0 + cut, c0, r0 + cut + INNER_WALL, c1))
            # A wall ending inside an existing doorway would split it into
            # two narrow gaps, so such cut positions are rejected.
            padded = (wall[0] - 1, wall[1] - 1, wall[2] + 1, wall[3] + 1)
            if not touches_door(*padded):
                break
        else:
            return
        grid[wall[0]:wall[2], wall[1]:wall[3]] = WALL
        walls.append((wall, vertical))
        length = rows if vertical else columns
        size = rng.randint(*DOOR)
        start = rng.randint(DOOR_END_GAP, length - size - DOOR_END_GAP)
        door = ((r0 + start, wall[1], r0 + start + size, wall[3]) if vertical
                else (wall[0], c0 + start, wall[2], c0 + start + size))
        grid[door[0]:door[2], door[1]:door[3]] = FREE
        doors.append((door[0] - DOOR_PAD, door[1] - DOOR_PAD, door[2] + DOOR_PAD, door[3] + DOOR_PAD))
        if vertical:
            split(r0, c0, r1, c0 + cut, depth + 1)
            split(r0, c0 + cut + INNER_WALL, r1, c1, depth + 1)
        else:
            split(r0, c0, r0 + cut, c1, depth + 1)
            split(r0 + cut + INNER_WALL, c0, r1, c1, depth + 1)

    r0, c0 = top + OUTER_WALL, left + OUTER_WALL
    r1, c1 = top + height - OUTER_WALL, left + width - OUTER_WALL
    if not courtyard:
        split(r0, c0, r1, c1, 0)
    else:
        # Each side band of the ring is at least one room deep, so the ring
        # stays passable; bands are split like the rest of the building, and
        # every split wall gets a door, so the way round is never closed.
        rows, columns = r1 - r0, c1 - c0
        yard_rows = rng.randint(MIN_COURTYARD, rows - 2 * MIN_ROOM)
        yard_columns = rng.randint(MIN_COURTYARD, columns - 2 * MIN_ROOM)
        y0 = r0 + rng.randint(MIN_ROOM, rows - MIN_ROOM - yard_rows)
        x0 = c0 + rng.randint(MIN_ROOM, columns - MIN_ROOM - yard_columns)
        y1, x1 = y0 + yard_rows, x0 + yard_columns
        grid[y0:y1, x0:x1] = WALL
        for band in ((r0, c0, y0, c1), (y1, c0, r1, c1), (y0, c0, y1, x0), (y0, x1, y1, c1)):
            split(*band, 1)

    # Extra doors turn the tree of rooms into a graph with loops. A door is
    # only cut where the wall has free space on both sides along its whole
    # width, so it never opens into a junction or the outside.
    added = 0
    for _ in range(200 if walls else 0):
        if added == extra_doors:
            break
        (r0, c0, r1, c1), vertical = rng.choice(walls)
        length = (r1 - r0) if vertical else (c1 - c0)
        size = rng.randint(*DOOR)
        if length < size + 2 * DOOR_END_GAP:
            continue
        start = rng.randint(DOOR_END_GAP, length - size - DOOR_END_GAP)
        if vertical:
            cells = (slice(r0 + start, r0 + start + size), slice(c0, c1))
            sides = grid[cells[0], c0 - 1], grid[cells[0], c1]
        else:
            cells = (slice(r0, r1), slice(c0 + start, c0 + start + size))
            sides = grid[r0 - 1, cells[1]], grid[r1, cells[1]]
        if np.all(grid[cells] == WALL) and all(np.all(side == FREE) for side in sides):
            grid[cells] = FREE
            added += 1
    # Each wall splits one region in two. The courtyard ring starts as one
    # region that its first wall only opens, hence one fewer room. Walls that
    # end in the open part of a ring corner add none, so courtyard counts are
    # approximate.
    return grid, max(1, len(walls) + (0 if courtyard else 1)), added


def components(mask):
    """Count 4-connected components, the same neighbours Cost's distance map floods."""
    seen, count = np.zeros_like(mask, dtype=bool), 0
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        queue = deque([start])
        while queue:
            row, column = queue.popleft()
            for step_row, step_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nearby = (row + step_row, column + step_column)
                if 0 <= nearby[0] < mask.shape[0] and 0 <= nearby[1] < mask.shape[1] \
                        and mask[nearby] and not seen[nearby]:
                    seen[nearby] = True
                    queue.append(nearby)
    return count


def erode(mask, radius):
    """Keep cells whose whole (2r+1)-square neighbourhood is free."""
    padded = np.pad(mask, radius, constant_values=False)
    window = np.lib.stride_tricks.sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    return window.all(axis=(2, 3))


def validate(path, seed, smoke_steps):
    """Return (check name, passed, detail) for one .pgm file on disk."""
    from explore_bench.config import configuration, task_for
    grid = np.asarray(Image.open(path))
    free = grid == FREE
    checks = []
    checks.append(("format", grid.shape == (SIZE, SIZE) and grid.dtype == np.uint8
                   and set(np.unique(grid)) <= {WALL, OUTSIDE, FREE}, "{} {}".format(grid.shape, grid.dtype)))
    outside = np.pad(grid == OUTSIDE, 1, constant_values=True)
    leaks = free & (outside[:-2, 1:-1] | outside[2:, 1:-1] | outside[1:-1, :-2] | outside[1:-1, 2:])
    checks.append(("enclosed", not leaks.any(), "{} free cells touch outside".format(int(leaks.sum()))))
    pieces = components(free)
    checks.append(("connected", pieces == 1, "{} free region(s)".format(pieces)))
    wide = components(erode(free, (MIN_PASSAGE - 1) // 2))
    checks.append(("passages >= {:.1f} m".format(MIN_PASSAGE / 10), wide == 1,
                   "{} region(s) after removing narrow gaps".format(wide)))
    area = free.sum() * 0.01
    checks.append(("free area", AREA_M2[0] <= area <= AREA_M2[1], "{:.0f} m²".format(area)))
    # The real pipeline: task_for re-checks the format and samples spawns
    # exactly as `prepare` does; Session then loads the file and runs a few
    # Cost decisions, as `evaluate` would.
    try:
        spec = configuration()
        task = task_for(spec, str(path), seed, "generated")
        detail = "spawns {}".format(task.start_positions)
        if smoke_steps:
            from explore_bench.controllers import ClassicalController
            from explore_bench.evaluation import baseline_view
            from explore_bench.session import Session
            session = Session(spec)
            controller = ClassicalController("cost", None)
            controller.bind_view = lambda: baseline_view(session)
            actor, _, available = session.reset_task(task)
            for _ in range(smoke_steps):
                request, _ = controller.act(actor, None, available)
                (actor, _, available), _, _, _, measurement = session.step_request(request)
            detail += ", {} Cost steps, coverage {:.1%}".format(smoke_steps, measurement["team_coverage_ratio"])
        checks.append(("benchmark loads it", True, detail))
    except Exception as error:  # report, don't crash: the point is to see failures
        checks.append(("benchmark loads it", False, repr(error)))
    return checks


def show(results, preview):
    import matplotlib
    if preview:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    columns = min(3, len(results))
    rows = -(-len(results) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5.4 * rows), squeeze=False)
    colors = ListedColormap(["#26313d", "#e1e0d9", "#ffffff"])
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, grid, rooms, loops, checks) in zip(axes.flat, results):
        ax.imshow(np.select([grid == WALL, grid == FREE], [0, 2], 1), cmap=colors, vmin=0, vmax=2,
                  interpolation="nearest")
        failed = [check for check, passed, _ in checks if not passed]
        verdict = "FAIL: " + ", ".join(failed) if failed else "PASS ({} checks)".format(len(checks))
        area = dict((check, detail) for check, _, detail in checks)["free area"]
        ax.set_title("{}  {}\n{} rooms, {} extra doors, {} free".format(name, verdict, rooms, loops, area),
                     loc="left", fontsize=11)
    figure.suptitle("Generated room-and-door maps (temporary; deleted after this window closes). "
                    "Details per check are printed in the terminal.", x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    if preview:
        figure.savefig(preview, dpi=120)
    else:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1, help="map i uses seed + i, so runs are reproducible")
    parser.add_argument("--extra-doors", type=int, default=0, help="doors added after splitting, creating loops")
    parser.add_argument("--courtyard", action="store_true", help="build round a central block (one large loop)")
    parser.add_argument("--smoke-steps", type=int, default=3, help="Cost decisions to run per map (0 skips)")
    parser.add_argument("--preview", type=Path, help="save the figure here instead of opening a window")
    args = parser.parse_args(argv)
    results = []
    with tempfile.TemporaryDirectory(prefix="explore_bench_maps_") as directory:
        for index in range(args.count):
            seed = args.seed + index
            grid, rooms, loops = generate(random.Random(seed), args.extra_doors, courtyard=args.courtyard)
            name = "rooms_seed_{}".format(seed)
            path = Path(directory) / (name + ".pgm")
            Image.fromarray(grid).save(path)
            checks = validate(path, seed, args.smoke_steps)
            results.append((name, np.asarray(Image.open(path)), rooms, loops, checks))
            print("{}: {} rooms, {} extra doors".format(name, rooms, loops))
            for check, passed, detail in checks:
                print("  {} {:<22} {}".format("ok  " if passed else "FAIL", check, detail))
        show(results, args.preview)
    print("Deleted", directory)
    return 0 if all(passed for *_, checks in results for _, passed, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
