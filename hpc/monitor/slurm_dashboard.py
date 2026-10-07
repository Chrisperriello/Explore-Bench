#!/usr/bin/env python3
"""Serve a live browser dashboard for one Level-0 Slurm collection."""

import argparse
import csv
import datetime as datetime_module
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

from slurm_monitor import cpu_estimate, query_queue, query_usage, read_last_episode, tail


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SLURM_ROOT = REPOSITORY_ROOT.parent / "slurm"
COLLECTION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
LOG_JOB_PATTERN = re.compile(r"-(\d+(?:_\d+)?)\.(?:out|err)$")
VARIANTS = ("original", "four_beam")


def utc_now():
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def load_tasks(path):
    """Load the frozen task table keyed by its integer array index."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["task_id"]): row for row in csv.DictReader(handle)}


def load_statuses(data_root, variant):
    """Load one status file per logical task for a sensor variant."""
    statuses = {}
    variant_root = data_root / variant
    if not variant_root.is_dir():
        return statuses
    for path in variant_root.glob("task_*/task_status.json"):
        status = read_json(path)
        task = status.get("task", {})
        try:
            task_id = int(task.get("task_id", path.parent.name.split("_")[-1]))
        except (TypeError, ValueError):
            continue
        status["_status_path"] = str(path)
        statuses[task_id] = status
    return statuses


def index_logs(log_root):
    """Index nested array logs once instead of scanning them for every row."""
    logs = {}
    if not log_root.is_dir():
        return logs
    for path in log_root.rglob("*"):
        if not path.is_file():
            continue
        match = LOG_JOB_PATTERN.search(path.name)
        if match is None:
            continue
        entry = logs.setdefault(match.group(1), {"stdout": "", "stderr": ""})
        if path.suffix == ".out":
            entry["stdout"] = str(path)
        elif path.suffix == ".err":
            entry["stderr"] = str(path)
    return logs


def array_parent(job_id):
    return str(job_id).split("_", 1)[0]


def logical_job_id(status, fallback_task_id, inferred_array_id):
    array_id = str(status.get("slurm_array_job_id") or inferred_array_id or "")
    task_id = str(status.get("slurm_array_task_id") or fallback_task_id)
    return "{}_{}".format(array_id, task_id) if array_id else ""


def infer_array_ids(statuses_by_variant):
    result = {}
    for variant, statuses in statuses_by_variant.items():
        identifiers = {
            str(status.get("slurm_array_job_id"))
            for status in statuses.values()
            if status.get("slurm_array_job_id")
        }
        if len(identifiers) == 1:
            result[variant] = identifiers.pop()
    return result


def elapsed_from_status(status):
    """Calculate completed wall time from collector timestamps on Python 3.6."""
    start = status.get("started_at")
    end = status.get("completed_at")
    if not start or not end:
        return ""

    def parse(value):
        value = str(value).replace("Z", "+00:00")
        value = value.split("+", 1)[0]
        formats = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")
        for date_format in formats:
            try:
                return datetime_module.datetime.strptime(value, date_format)
            except ValueError:
                pass
        return None

    started = parse(start)
    completed = parse(end)
    if started is None or completed is None:
        return ""
    seconds = max(0, int((completed - started).total_seconds()))
    return "{:02d}:{:02d}:{:02d}".format(seconds // 3600, (seconds // 60) % 60, seconds % 60)


def task_run_directory(task_directory, status):
    run_text = str(status.get("run_directory") or "")
    if run_text:
        return Path(run_text)
    manifests = list(task_directory.glob("*/run*/manifest.json"))
    if not manifests:
        return None
    return max(manifests, key=lambda path: path.stat().st_mtime).parent


def task_state(data_status, queue_state):
    if data_status == "complete":
        return "COMPLETED", "complete"
    if data_status == "failed":
        return "FAILED", "failed"
    queue_state = queue_state.upper()
    if queue_state in ("RUNNING", "COMPLETING"):
        return queue_state, "running"
    if queue_state in ("PENDING", "CONFIGURING"):
        return queue_state, "waiting"
    if data_status == "running":
        return "STARTING", "running"
    return "PENDING", "waiting"


def build_snapshot(
    slurm_root,
    collection,
    user,
    queue_provider=query_queue,
    usage_provider=query_usage,
):
    """Build one exact task-level view without querying completed-job history."""
    task_file = slurm_root / "tasks" / "level0" / collection / "tasks.csv"
    if not task_file.is_file():
        raise ValueError("task table not found: {}".format(task_file))
    tasks = load_tasks(task_file)
    data_root = slurm_root / "data" / "level0" / collection
    log_root = slurm_root / "logs" / "level0" / collection
    statuses_by_variant = {
        variant: load_statuses(data_root, variant) for variant in VARIANTS
    }
    array_ids = infer_array_ids(statuses_by_variant)

    queue, queue_error = queue_provider(user)
    if array_ids:
        allowed = set(array_ids.values())
        queue = {
            job_id: job
            for job_id, job in queue.items()
            if array_parent(job_id) in allowed
        }
    else:
        queue = {}
    usage, usage_error = usage_provider(queue)
    logs = index_logs(log_root)

    rows = []
    for variant in VARIANTS:
        statuses = statuses_by_variant[variant]
        variant_array_id = array_ids.get(variant, "")
        for task_id in sorted(tasks):
            task = tasks[task_id]
            status = statuses.get(task_id, {})
            job_id = logical_job_id(status, task_id, variant_array_id)
            job = queue.get(job_id, {})
            state, health = task_state(
                str(status.get("status", "")).lower(), str(job.get("state", ""))
            )
            task_directory = data_root / variant / "task_{:06d}".format(task_id)
            run_directory = task_run_directory(task_directory, status)
            episode = (
                read_last_episode(run_directory / "episodes.csv")
                if run_directory is not None
                else {}
            )
            task_logs = logs.get(job_id, {"stdout": "", "stderr": ""})
            error_lines = tail(task_logs.get("stderr", ""), lines=4)
            elapsed = str(job.get("elapsed", "")) or elapsed_from_status(status)
            rows.append(
                {
                    "task_id": task_id,
                    "job_id": job_id,
                    "variant": variant,
                    "sensor": "360-degree" if variant == "original" else "four-beam",
                    "method": task.get("method", ""),
                    "map": task.get("map_name", ""),
                    "seed": task.get("seed", ""),
                    "state": state,
                    "health": health,
                    "elapsed": elapsed,
                    "cpu": cpu_estimate(job, usage.get(job_id, {})) if job else "-",
                    "memory": usage.get(job_id, {}).get("maximum_rss")
                    or usage.get(job_id, {}).get("average_rss")
                    or "-",
                    "node": job.get("reason") or job.get("nodes", ""),
                    "coverage": episode.get("final_coverage_ratio", ""),
                    "steps": episode.get("steps_executed", ""),
                    "termination": episode.get("termination_reason", ""),
                    "run_directory": str(run_directory) if run_directory else "",
                    "stdout": task_logs.get("stdout", ""),
                    "stderr": task_logs.get("stderr", ""),
                    "error_tail": "".join(error_lines).strip(),
                }
            )

    groups = {}
    totals = {"complete": 0, "running": 0, "waiting": 0, "failed": 0}
    for variant in VARIANTS:
        group_rows = [row for row in rows if row["variant"] == variant]
        counts = {
            name: sum(row["health"] == name for row in group_rows)
            for name in totals
        }
        counts["total"] = len(group_rows)
        counts["array_job_id"] = array_ids.get(variant, "")
        groups[variant] = counts
        for name in totals:
            totals[name] += counts[name]
    totals["total"] = len(rows)

    errors = [value for value in (queue_error, usage_error) if value]
    return {
        "collection": collection,
        "generated_at": utc_now(),
        "groups": groups,
        "totals": totals,
        "rows": rows,
        "errors": errors,
    }


class SnapshotCache:
    def __init__(self, builder, minimum_age):
        self.builder = builder
        self.minimum_age = minimum_age
        self.lock = threading.Lock()
        self.created = 0.0
        self.value = None

    def get(self):
        with self.lock:
            if self.value is None or time.time() - self.created >= self.minimum_age:
                try:
                    self.value = self.builder()
                except Exception as error:
                    self.value = {
                        "collection": "",
                        "generated_at": utc_now(),
                        "groups": {},
                        "totals": {},
                        "rows": [],
                        "errors": ["{}: {}".format(type(error).__name__, error)],
                    }
                self.created = time.time()
            return self.value


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def handler_class(cache, refresh_seconds):
    class DashboardHandler(BaseHTTPRequestHandler):
        def send_bytes(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                page = DASHBOARD_HTML.replace(
                    "__REFRESH_MILLISECONDS__", str(int(refresh_seconds * 1000))
                )
                self.send_bytes(200, "text/html; charset=utf-8", page.encode("utf-8"))
            elif path == "/api/snapshot":
                payload = json.dumps(cache.get(), separators=(",", ":")).encode("utf-8")
                self.send_bytes(200, "application/json; charset=utf-8", payload)
            elif path == "/healthz":
                self.send_bytes(200, "text/plain; charset=utf-8", b"ok\n")
            elif path == "/favicon.ico":
                self.send_bytes(204, "image/x-icon", b"")
            else:
                self.send_bytes(404, "text/plain; charset=utf-8", b"not found\n")

        def log_message(self, message_format, *arguments):
            return

    return DashboardHandler


DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explore-Bench Level-0</title>
<style>
:root{color-scheme:dark;--bg:#0a0d12;--panel:#121720;--panel2:#181f2a;--line:#273140;--text:#edf3f8;--muted:#91a0b2;--blue:#58a6ff;--cyan:#39d0d8;--green:#4bd68a;--amber:#f5b84b;--red:#ff6b78}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -10%,#172843 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.shell{max-width:1500px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:22px}.eyebrow{color:var(--cyan);font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}h1{font-size:30px;margin:5px 0 3px;letter-spacing:-.03em}.muted{color:var(--muted)}.live{display:flex;align-items:center;gap:8px;border:1px solid var(--line);background:#10161e;border-radius:999px;padding:8px 12px}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green)}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-bottom:16px}.card,.table-card{background:linear-gradient(145deg,rgba(24,31,42,.96),rgba(16,21,29,.96));border:1px solid var(--line);border-radius:14px;box-shadow:0 14px 38px rgba(0,0,0,.22)}.card{padding:18px}.card-head{display:flex;justify-content:space-between;align-items:center}.card h2{font-size:16px;margin:0}.job{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}.big{font-size:28px;font-weight:750;margin-top:14px}.progress{height:8px;background:#252e3b;border-radius:8px;overflow:hidden;margin:10px 0 13px}.bar{height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));transition:width .35s}.counts{display:flex;gap:16px;flex-wrap:wrap;font-size:12px}.count strong{font-size:14px;margin-right:4px}.green{color:var(--green)}.amber{color:var(--amber)}.red{color:var(--red)}.blue{color:var(--blue)}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:14px 15px;border-bottom:1px solid var(--line)}input,select{background:#0d1219;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;outline:none}input{min-width:260px}input:focus,select:focus{border-color:var(--blue)}.spacer{flex:1}.table-wrap{overflow:auto;max-height:62vh}table{width:100%;border-collapse:collapse;white-space:nowrap}th{position:sticky;top:0;background:#151b24;color:#9fadc0;text-align:left;text-transform:uppercase;letter-spacing:.08em;font-size:10px;padding:11px 12px;border-bottom:1px solid var(--line)}td{padding:10px 12px;border-bottom:1px solid rgba(39,49,64,.65)}tr:hover td{background:rgba(88,166,255,.045)}.badge{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:750;text-transform:uppercase}.badge.complete{background:rgba(75,214,138,.13);color:var(--green)}.badge.running{background:rgba(88,166,255,.14);color:var(--blue)}.badge.waiting{background:rgba(245,184,75,.13);color:var(--amber)}.badge.failed{background:rgba(255,107,120,.13);color:var(--red)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.errors{display:none;margin-bottom:16px;padding:12px 14px;border:1px solid rgba(255,107,120,.4);background:rgba(255,107,120,.08);border-radius:10px;color:#ffb1b8}.empty{text-align:center;color:var(--muted);padding:42px}.footer{color:var(--muted);font-size:12px;margin-top:12px;text-align:right}@media(max-width:760px){.shell{padding:16px}.cards{grid-template-columns:1fr}.top{flex-direction:column}input{width:100%;min-width:0}.table-wrap{max-height:58vh}}
</style>
</head>
<body><main class="shell">
<div class="top"><div><div class="eyebrow">Live cluster telemetry</div><h1>Explore-Bench Level-0</h1><div class="muted" id="subtitle">Connecting…</div></div><div class="live"><span class="dot"></span><span id="freshness">Loading</span></div></div>
<div id="errors" class="errors"></div><section id="cards" class="cards"></section>
<section class="table-card"><div class="toolbar"><input id="search" placeholder="Filter task, map, method, seed…"><select id="state"><option value="active">Active + failed</option><option value="all">All tasks</option><option value="running">Running</option><option value="waiting">Waiting</option><option value="complete">Complete</option><option value="failed">Failed</option></select><div class="spacer"></div><span class="muted" id="shown"></span></div><div class="table-wrap"><table><thead><tr><th>Task</th><th>Sensor</th><th>Method</th><th>Map</th><th>Seed</th><th>Status</th><th>Time</th><th>CPU</th><th>Memory</th><th>Coverage</th><th>Steps</th><th>Stopped by</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<div class="footer">Auto-refreshing from Ada · <span id="updated">—</span></div>
</main><script>
const refreshMs=__REFRESH_MILLISECONDS__;let snapshot=null;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>{const n=Number(v);return Number.isFinite(n)&&v!==''?(n*100).toFixed(1)+'%':'—'};
function card(key,label){const g=snapshot.groups[key]||{total:0,complete:0,running:0,waiting:0,failed:0};const p=g.total?100*g.complete/g.total:0;return `<article class="card"><div class="card-head"><h2>${label}</h2><span class="job">array ${esc(g.array_job_id||'discovering')}</span></div><div class="big">${g.complete} <span class="muted">/ ${g.total}</span></div><div class="progress"><div class="bar" style="width:${p}%"></div></div><div class="counts"><span class="count green"><strong>${g.complete}</strong> complete</span><span class="count blue"><strong>${g.running}</strong> running</span><span class="count amber"><strong>${g.waiting}</strong> waiting</span><span class="count red"><strong>${g.failed}</strong> failed</span></div></article>`}
function render(){if(!snapshot)return;document.getElementById('subtitle').textContent=`Collection ${snapshot.collection} · ${snapshot.totals.total||0} matched runs`;document.getElementById('cards').innerHTML=card('original','Original 360°')+card('four_beam','Four-beam');const errors=document.getElementById('errors');errors.style.display=snapshot.errors.length?'block':'none';errors.textContent=snapshot.errors.join(' · ');const q=document.getElementById('search').value.toLowerCase();const mode=document.getElementById('state').value;let rows=snapshot.rows.filter(r=>{const text=[r.task_id,r.job_id,r.sensor,r.method,r.map,r.seed,r.state].join(' ').toLowerCase();const stateOk=mode==='all'||(mode==='active'&&r.health!=='complete')||r.health===mode;return stateOk&&text.includes(q)});const order={failed:0,running:1,waiting:2,complete:3};rows.sort((a,b)=>(order[a.health]-order[b.health])||a.variant.localeCompare(b.variant)||a.task_id-b.task_id);document.getElementById('shown').textContent=`${rows.length} shown`;document.getElementById('rows').innerHTML=rows.length?rows.map(r=>`<tr title="${esc(r.error_tail||r.run_directory)}"><td class="mono">${String(r.task_id).padStart(3,'0')}</td><td>${esc(r.sensor)}</td><td>${esc(r.method)}</td><td>${esc(r.map)}</td><td>${esc(r.seed)}</td><td><span class="badge ${r.health}">${esc(r.state)}</span></td><td class="mono">${esc(r.elapsed||'—')}</td><td>${esc(r.cpu)}</td><td>${esc(r.memory)}</td><td>${pct(r.coverage)}</td><td>${esc(r.steps||'—')}</td><td>${esc(r.termination||'—')}</td></tr>`).join(''):'<tr><td colspan="12" class="empty">No tasks match this filter.</td></tr>';document.getElementById('updated').textContent=new Date(snapshot.generated_at).toLocaleTimeString();document.getElementById('freshness').textContent='Live'}
async function load(){try{const response=await fetch('/api/snapshot',{cache:'no-store'});snapshot=await response.json();render()}catch(error){document.getElementById('freshness').textContent='Disconnected';document.querySelector('.dot').style.background='var(--red)'}}
document.getElementById('search').addEventListener('input',render);document.getElementById('state').addEventListener('change',render);load();setInterval(load,refreshMs);
</script></body></html>'''


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_SLURM_ROOT)
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=float, default=5.0)
    return parser


def main():
    args = build_parser().parse_args()
    if not COLLECTION_PATTERN.match(args.collection):
        raise SystemExit("invalid collection ID")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.refresh < 1.0:
        raise SystemExit("--refresh must be at least one second")
    args.root = args.root.expanduser().resolve()
    cache = SnapshotCache(
        lambda: build_snapshot(args.root, args.collection, args.user), args.refresh
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_class(cache, args.refresh)
    )
    print("Explore-Bench dashboard: http://{}:{}".format(args.host, args.port))
    print("Collection: {}".format(args.collection))
    print("Press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
