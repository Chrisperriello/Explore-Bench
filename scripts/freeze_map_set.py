"""Freeze a set of generated room maps, with a manifest, for experiments to use.

Usage: python scripts/freeze_map_set.py --name rooms --version 1 [--splits test=30]
                                        [--seed 1] [--extra-doors 0 1 2 3]
                                        [--smoke-steps 3] [--root maps] [--allow-dirty]
       python scripts/freeze_map_set.py --check maps/rooms_v1

Maps come from scripts/generate_room_maps.py and must pass its validation;
a failing seed is skipped and recorded, and the next seed is tried. Map i
uses --extra-doors[i % n] extra doors, so every split gets the same mix of
tree-shaped and looped layouts. Each split is listed in the manifest with
paths relative to the repository, ready to paste into an experiment's "maps".

Sets are write-once: an existing <root>/<name>_v<version> is never touched;
a change becomes a new version. The set is built in a temporary directory
beside it and renamed into place only when every map passed, so a failed
run leaves nothing behind. Use --root results/maps for scratch sets.
"""

import argparse
import datetime
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_room_maps as rooms  # noqa: E402
from explore_bench.config import DATASETS  # noqa: E402
from explore_bench.map_tags import FREE, OUTSIDE, WALL, load, tags  # noqa: E402

SCHEMA = 1
SPLITS = ("train", "validation", "test")
# Uncommitted edits to these would make the recorded commit unable to
# reproduce the set.
SOURCES = ("scripts/freeze_map_set.py", "scripts/generate_room_maps.py", "explore_bench/map_tags.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    return commit, bool(dirty)


def parse_splits(values):
    counts = {}
    for value in values:
        split, _, count = value.partition("=")
        if split not in SPLITS or split in counts or not count.isdigit() or int(count) < 1:
            raise ValueError("splits look like test=30 train=100, each split once")
        counts[split] = int(count)
    return counts


def build(directory, set_name, counts, seed, extra_doors, smoke_steps):
    """Generate and validate maps into directory; return the manifest's map lists."""
    known = {sha256(path) for path in (ROOT / DATASETS).glob("*.pgm")}
    maps, rejected, index = [], [], 0
    next_seed = seed
    for split in SPLITS:
        for _ in range(counts.get(split, 0)):
            requested = extra_doors[index % len(extra_doors)]
            while True:
                map_seed, next_seed = next_seed, next_seed + 1
                grid, room_count, added = rooms.generate(random.Random(map_seed), requested)
                name = "{}_seed_{}".format(set_name, map_seed)
                path = directory / (name + ".pgm")
                Image.fromarray(grid).save(path)
                checks = rooms.validate(path, map_seed, smoke_steps)
                failed = [check for check, passed, _ in checks if not passed]
                digest = sha256(path)
                if digest in known:
                    failed.append("duplicate content")
                if not failed:
                    break
                rejected.append({"seed": map_seed, "failed": failed})
                path.unlink()
                if len(rejected) > 10 * sum(counts.values()):
                    raise RuntimeError("too many rejected maps; check the generator settings")
            known.add(digest)
            maps.append({"name": name, "file": path.name, "sha256": digest, "split": split,
                         "seed": map_seed, "rooms": room_count, "extra_doors_requested": requested,
                         "extra_doors_added": added, "tags": tags(load(path)),
                         "checks": {check: detail for check, _, detail in checks}})
            print("{} {:<10} {} rooms, {} extra doors, {} loops".format(
                name, split, room_count, added, maps[-1]["tags"]["loops"]), flush=True)
            index += 1
    return maps, rejected


def freeze(args):
    counts = parse_splits(args.splits)
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("uncommitted changes in {}; commit them or pass --allow-dirty".format(", ".join(SOURCES)))
    root = (ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    set_name = "{}_v{}".format(args.name, args.version)
    target = root / set_name
    if target.exists():
        raise SystemExit("{} already exists; sets are write-once, use a new --version".format(target))
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + set_name + ".", dir=root))
    try:
        maps, rejected = build(staging, set_name, counts, args.seed, args.extra_doors, args.smoke_steps)
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            relative = target
        manifest = {
            "schema": SCHEMA, "name": set_name, "source": "rooms",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "commit": commit, "sources_dirty": dirty,
            "parameters": {"seed": args.seed, "splits": counts, "extra_doors": args.extra_doors,
                           "smoke_steps": args.smoke_steps},
            "generator": {"size": rooms.SIZE, "margin": rooms.MARGIN, "min_room": rooms.MIN_ROOM,
                          "door": rooms.DOOR, "min_passage": rooms.MIN_PASSAGE, "area_m2": rooms.AREA_M2},
            # Enough to extrude each map into a Gazebo world later.
            "grid": {"resolution_m": 0.1, "free": FREE, "wall": WALL, "outside": OUTSIDE},
            "attempts": len(maps) + len(rejected), "rejected": rejected,
            "splits": {split: [str(relative / item["file"]) for item in maps if item["split"] == split]
                       for split in SPLITS if split in counts},
            "maps": maps,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.chmod(0o755)  # mkdtemp makes it owner-only
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("Froze {} maps ({} rejected) into {}".format(len(maps), len(rejected), target))
    return target


def check(directory):
    """Confirm a frozen set on disk still matches its manifest exactly."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {item["file"]: item["sha256"] for item in manifest["maps"]}
    present = {path.name for path in directory.glob("*.pgm")}
    problems = ["missing " + name for name in sorted(set(expected) - present)]
    problems += ["not in manifest " + name for name in sorted(present - set(expected))]
    problems += ["changed " + name for name in sorted(set(expected) & present)
                 if sha256(directory / name) != expected[name]]
    for problem in problems:
        print(problem)
    print("{}: {}".format(directory, "OK, {} maps".format(len(expected)) if not problems else "FAILED"))
    return not problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path, help="verify an existing set against its manifest")
    parser.add_argument("--name", default="rooms")
    parser.add_argument("--version", type=int)
    parser.add_argument("--splits", nargs="+", default=["test=30"])
    parser.add_argument("--seed", type=int, default=1, help="first generator seed; rejected seeds are skipped")
    parser.add_argument("--extra-doors", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--smoke-steps", type=int, default=3, help="Cost decisions run per map (0 skips)")
    parser.add_argument("--root", default="maps", help="parent directory, relative to the repository")
    parser.add_argument("--allow-dirty", action="store_true", help="for scratch sets only")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check(args.check) else 1
    if args.version is None or args.version < 1:
        parser.error("--version N (N >= 1) is required when freezing")
    if any(value < 0 for value in args.extra_doors):
        parser.error("--extra-doors values must be nonnegative")
    freeze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
