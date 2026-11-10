"""Freeze a set of generated room maps or HouseExpo layouts, with a manifest.

Usage: python scripts/freeze_map_set.py --name rooms --version 1 [--splits test=30]
                                        [--seed 1] [--extra-doors 0 1 2 3] [--courtyards 0 1]
                                        [--smoke-steps 3] [--root maps] [--allow-dirty]
       python scripts/freeze_map_set.py --source houseexpo --name houseexpo --version 1
                                        [--archive json.tar.gz] [--max-attempts 400] ...
       python scripts/freeze_map_set.py --check maps/rooms_v1
       python scripts/freeze_map_set.py --rebuild maps/houseexpo_v1 [--archive json.tar.gz]

Rooms come from scripts/generate_room_maps.py. Map i uses extra doors
--extra-doors[i % n] and, cycling more slowly, courtyard --courtyards[...],
so every split gets the same mix of tree-shaped, small-loop and large-loop
layouts. HouseExpo layouts are sampled from the pinned archive and converted
by scripts/preview_houseexpo_maps.py. Either way a map must pass its source's
validation; a failing candidate is skipped and recorded.

Sets are write-once: an existing <root>/<name>_v<version> is never touched;
a change becomes a new version. The set is built in a temporary directory
beside it and renamed into place only when every map passed, so a failed
run leaves nothing behind. Use --root results/maps for scratch sets.

HouseExpo is built on SUNCG, whose redistribution terms are unclear, so a
HouseExpo set commits only its manifest (a .gitignore inside it skips the
.pgm files). --rebuild recreates the maps from the archive on another
machine and refuses any map whose SHA-256 differs from the manifest.
"""

import argparse
import datetime
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_room_maps as rooms  # noqa: E402
import preview_houseexpo_maps as houseexpo  # noqa: E402
from explore_bench.config import DATASETS  # noqa: E402
from explore_bench.map_tags import FREE, OUTSIDE, WALL, load, tags  # noqa: E402

SCHEMA = 1
SPLITS = ("train", "validation", "test")
# Uncommitted edits to these would make the recorded commit unable to
# reproduce the set.
SOURCES = ("scripts/freeze_map_set.py", "scripts/generate_room_maps.py",
           "scripts/preview_houseexpo_maps.py", "explore_bench/map_tags.py")


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


def houseexpo_seed(source_id, seed):
    """The preview's spawn seed, tied to the layout rather than its position in the sample."""
    return int(hashlib.sha256((source_id + ":" + str(seed)).encode()).hexdigest()[:8], 16)


def room_candidates(args):
    """Yield (name, grid, facts, validate) forever; the caller passes the map index in."""
    seed = args.seed
    index = yield
    while True:
        extra = args.extra_doors[index % len(args.extra_doors)]
        courtyard = bool(args.courtyards[(index // len(args.extra_doors)) % len(args.courtyards)])
        map_seed, seed = seed, seed + 1
        grid, room_count, added = rooms.generate(random.Random(map_seed), extra, courtyard=courtyard)
        facts = {"seed": map_seed, "rooms": room_count, "courtyard": courtyard,
                 "extra_doors_requested": extra, "extra_doors_added": added}
        index = yield ("{}_seed_{}".format(args.set_name, map_seed), grid, facts,
                       lambda path, s=map_seed: rooms.validate(path, s, args.smoke_steps))


def houseexpo_candidates(args):
    index = yield
    for source_id, payload in houseexpo.candidates(args.archive_path, args.seed, args.max_attempts):
        try:
            house = json.loads(payload)
            grid = houseexpo.rasterize(house)
        except (ValueError, TypeError, KeyError) as error:
            index = yield (source_id, None, {"source_id": source_id}, str(error))
            continue
        seed = houseexpo_seed(source_id, args.seed)
        facts = {"source_id": source_id, "rooms": house.get("room_num"), "spawn_check_seed": seed}
        index = yield ("{}_{}".format(args.set_name, source_id), grid, facts,
                       lambda path, s=seed: houseexpo.validate(path, s, args.smoke_steps)[0])


def build(directory, counts, candidates):
    """Validate candidates into directory until every split is full; return (maps, rejected)."""
    known = {sha256(path) for path in (ROOT / DATASETS).glob("*.pgm")}
    maps, rejected = [], []
    next(candidates)
    for split in SPLITS:
        for _ in range(counts.get(split, 0)):
            while True:
                try:
                    name, grid, facts, validate = candidates.send(len(maps))
                except StopIteration:
                    raise RuntimeError("source ran out after {} maps ({} rejected); raise --max-attempts"
                                       .format(len(maps), len(rejected)))
                if grid is None:
                    rejected.append(dict(facts, failed=[validate]))
                    continue
                path = directory / (name + ".pgm")
                Image.fromarray(grid).save(path)
                checks = validate(path)
                failed = [check for check, passed, _ in checks if not passed]
                digest = sha256(path)
                if digest in known:
                    failed.append("duplicate content")
                if not failed:
                    break
                rejected.append(dict(facts, failed=failed))
                path.unlink()
                if len(rejected) > 20 * sum(counts.values()):
                    raise RuntimeError("too many rejected maps; check the source settings")
            known.add(digest)
            maps.append(dict({"name": name, "file": path.name, "sha256": digest, "split": split},
                             **facts, tags=tags(load(path)),
                             checks={check: detail for check, _, detail in checks}))
            print("{:<40} {:<10} {} loops, largest island {} m²".format(
                name, split, maps[-1]["tags"]["loops"], maps[-1]["tags"]["largest_island_m2"]), flush=True)
    return maps, rejected


def archive(args, scratch):
    """Return a verified HouseExpo archive path, downloading into scratch if none is given."""
    if args.archive is None:
        return houseexpo.download_archive(scratch)
    if sha256(args.archive) != houseexpo.ARCHIVE_SHA256:
        raise SystemExit("{} is not the pinned HouseExpo archive".format(args.archive))
    return args.archive


def freeze(args):
    counts = parse_splits(args.splits)
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("uncommitted changes in {}; commit them or pass --allow-dirty".format(", ".join(SOURCES)))
    root = (ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    args.set_name = "{}_v{}".format(args.name, args.version)
    target = root / args.set_name
    if target.exists():
        raise SystemExit("{} already exists; sets are write-once, use a new --version".format(target))
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.set_name + ".", dir=root))
    try:
        parameters = {"seed": args.seed, "splits": counts, "smoke_steps": args.smoke_steps}
        with tempfile.TemporaryDirectory(prefix="explore_bench_freeze_") as scratch:
            if args.source == "rooms":
                parameters.update(extra_doors=args.extra_doors, courtyards=args.courtyards)
                generator = {"size": rooms.SIZE, "margin": rooms.MARGIN, "min_room": rooms.MIN_ROOM,
                             "door": rooms.DOOR, "min_passage": rooms.MIN_PASSAGE,
                             "min_courtyard": rooms.MIN_COURTYARD, "area_m2": rooms.AREA_M2}
                candidates = room_candidates(args)
            else:
                args.archive_path = archive(args, Path(scratch))
                parameters.update(max_attempts=args.max_attempts)
                generator = {"archive_url": houseexpo.ARCHIVE_URL, "archive_sha256": houseexpo.ARCHIVE_SHA256,
                             "supersample": houseexpo.SUPERSAMPLE, "area_m2": houseexpo.AREA_M2,
                             "clearance_cells": houseexpo.CLEARANCE_CELLS}
                candidates = houseexpo_candidates(args)
            maps, rejected = build(staging, counts, candidates)
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            relative = target
        manifest = {
            "schema": SCHEMA, "name": args.set_name, "source": args.source,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "commit": commit, "sources_dirty": dirty, "parameters": parameters, "generator": generator,
            # Enough to extrude each map into a Gazebo world later.
            "grid": {"resolution_m": 0.1, "free": FREE, "wall": WALL, "outside": OUTSIDE},
            "attempts": len(maps) + len(rejected), "rejected": rejected,
            "splits": {split: [str(relative / item["file"]) for item in maps if item["split"] == split]
                       for split in SPLITS if split in counts},
            "maps": maps,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.source == "houseexpo":
            (staging / ".gitignore").write_text("# SUNCG-derived; rebuild with freeze_map_set.py --rebuild\n*.pgm\n")
        staging.chmod(0o755)  # mkdtemp makes it owner-only
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("Froze {} maps ({} rejected) into {}".format(len(maps), len(rejected), target))
    return target


def rebuild(args):
    """Recreate a HouseExpo set's maps from the archive, exactly as recorded."""
    directory = Path(args.rebuild)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source"] != "houseexpo":
        raise SystemExit("only HouseExpo sets are rebuilt; room sets are committed with their maps")
    wanted = {item["source_id"]: item for item in manifest["maps"]}
    written = []
    try:
        with tempfile.TemporaryDirectory(prefix="explore_bench_rebuild_") as scratch:
            source = archive(args, Path(scratch))
            with tarfile.open(str(source), "r:gz") as tar:
                for member in tar:
                    item = wanted.get(Path(member.name).stem)
                    if item is None or not member.isfile() or not member.name.endswith(".json"):
                        continue
                    with tar.extractfile(member) as handle:
                        grid = houseexpo.rasterize(json.loads(handle.read()))
                    path = directory / item["file"]
                    Image.fromarray(grid).save(path)
                    written.append(path)
                    if sha256(path) != item["sha256"]:
                        raise SystemExit("{} differs from the manifest; check numpy/OpenCV/Pillow versions"
                                         .format(item["file"]))
        if len(written) != len(wanted):
            raise SystemExit("archive is missing {} layouts".format(len(wanted) - len(written)))
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return check(directory)


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
    if problems and manifest["source"] == "houseexpo" and not present:
        print("HouseExpo maps are not committed; run --rebuild {}".format(directory))
    print("{}: {}".format(directory, "OK, {} maps".format(len(expected)) if not problems else "FAILED"))
    return not problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path, help="verify an existing set against its manifest")
    parser.add_argument("--rebuild", type=Path, help="recreate a HouseExpo set's maps from the archive")
    parser.add_argument("--source", choices=("rooms", "houseexpo"), default="rooms")
    parser.add_argument("--name", default="rooms")
    parser.add_argument("--version", type=int)
    parser.add_argument("--splits", nargs="+", default=["test=30"])
    parser.add_argument("--seed", type=int, default=1, help="first room seed, or the HouseExpo sample seed")
    parser.add_argument("--extra-doors", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--courtyards", type=int, nargs="+", choices=(0, 1), default=[0],
                        help="room maps only; 0 1 alternates without and with a central block")
    parser.add_argument("--archive", type=Path, help="HouseExpo json.tar.gz; default: download into temporary storage")
    parser.add_argument("--max-attempts", type=int, default=400, help="HouseExpo layouts sampled at most")
    parser.add_argument("--smoke-steps", type=int, default=3, help="Cost decisions run per map (0 skips)")
    parser.add_argument("--root", default="maps", help="parent directory, relative to the repository")
    parser.add_argument("--allow-dirty", action="store_true", help="for scratch sets only")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check(args.check) else 1
    if args.rebuild:
        return 0 if rebuild(args) else 1
    if args.version is None or args.version < 1:
        parser.error("--version N (N >= 1) is required when freezing")
    if any(value < 0 for value in args.extra_doors):
        parser.error("--extra-doors values must be nonnegative")
    freeze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
