# Complete Level-0 setup

## One setup command

From the checkout, on Ada inside an existing compute allocation:

```bash
source ./setup.sh --hpc
```

For a new allocation first use:

```bash
srun --partition=ada --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=16G --time=02:00:00 --pty bash -l
cd ~/Explore-Bench
source ./setup.sh --hpc
```

Setup does the installation and checks; do not run separate pip, editable
install or pytest commands afterward. Source it to also clear conflicting
variables in the calling shell and put the project interpreter on PATH.
HPC activation also selects headless plotting, unbuffered output and one
thread per numeric-library process, matching the job environment.
`bash setup.sh --hpc` performs the same installation/checks but, like any child
process, cannot change its parent shell's `PYTHONHOME` or PATH.

For local development use `source ./setup.sh --local`. Linux defaults to `--hpc`
(`.conda-env`); macOS defaults to `--local` (`.venv`). The older
`source ./setup_env.sh` entry point delegates to the same implementation.
`--full` and `--editable` remain accepted but are now always enabled.

## What is provisioned and verified

1. Clear inherited `PYTHONHOME`, `PYTHONPATH` and user-site/install overrides.
   Ada's Anaconda module sets `PYTHONHOME` to its own newer Python; leaving it
   set makes the project Python fail before importing `encodings`.
2. Reuse a working project Python **3.8.20**, or create it. HPC uses Conda,
   optionally loaded from Ada's module. If no Conda/Python provider exists,
   download a repository-private, SHA-256-verified Miniforge 24.9.2-0 bootstrap
   under `.setup/`. Local venvs can also use an installed uv or Python 3.8.20.
   No global Python, shell startup file or system package is modified.
3. Install pinned pip/setuptools/wheel and the selected PyTorch/torchvision
   build. Linux defaults to the official **CPU-only** wheel index rather than
   the large CUDA dependency set selected by PyPI's Linux default.
4. Install the grid **and** MAPPO requirements. Use headless OpenCV to avoid
   system X11/libGL requirements on compute nodes. Replace conflicting OpenCV
   distributions only inside this project environment; do not install two
   distributions that both own `cv2`.
5. Install `onpolicy/` and the root `explore_bench` package in editable mode.
   Installing torch alone does not expose `onpolicy/onpolicy` as the Python
   package `onpolicy`. Editable installation is mandatory, not a PYTHONPATH
   workaround. Source changes remain visible to spawned worker processes.
6. Run `pip check`, isolated import/native-library checks, NumPy↔PyTorch tensor
   execution, default map/configuration checks and a CLI subprocess outside the
   checkout. Verify package origins point at this checkout.
7. Run the full regression suite, with individual test names, slow-test timings
   and stack dumps after 120 seconds of a quiet test. Print success only if all
   checks finish successfully. A stack dump diagnoses slowness; it is not a
   hard test timeout.

Linux dependency checks are strict. On macOS arm64 only, the pinned grpcio
1.70.0 CPython 3.8 universal2 wheel has an incorrect internal x86_64-only
`WHEEL` tag, although its native arm64 extension loads. That **single exact**
`pip check` platform warning is accepted only after checking the installed
version/tag and loading the native extension. It remains visible in the log
and JSON (`dependency_check.status=known_metadata_warning`). All dependency
conflicts, other platform warnings and native import failures still stop setup.
Third-party package metadata is not rewritten to hide the warning.

Setup uses the full Level-0 stack by default, not the legacy all-benchmark
requirements. ROS/Gazebo, hardware drivers, Atari, MuJoCo, StarCraft, MPI and
TensorFlow are outside this setup's scope. Python 3.8 is retained for this
legacy experiment; this is not a claim that its ecosystem is still supported
or appropriate for exposing services to untrusted networks.

## Clean-machine prerequisites

Supported: Linux x86_64 and macOS x86_64/arm64, a writable checkout, Bash and
ordinary OS tools, HTTPS access to PyPI/PyTorch/conda-forge/GitHub, and `curl`
or `wget` if bootstrapping a provider. The checksum step uses `sha256sum` or
`shasum`. A Git clone with a HEAD commit is required for provenance (a ZIP/tar
without Git history is rejected before installation); actual HPC execution requires the cluster's
existing Slurm installation and permission to allocate resources. Setup does
not install a scheduler, OS, GPU driver or firewall rules, and never uses sudo.

Keep enough disk space for downloads plus the environment (several GB); fresh
Conda provisioning and dependency downloads can take minutes. Installer output
and explicit import/test progress remain visible. The initial two-hour Slurm
allocation is a setup allowance, not a measured training runtime.

Only Python 3.8.20 is accepted. An existing differently versioned or incomplete
environment is not silently reused. For local creation from a specific
interpreter use `--local --python /absolute/path/to/python3.8`.

## Logs, failure and retry

```text
.setup/setup_<UTC timestamp>_<PID>.log
.setup/environment_check.json
```

The JSON records the actual interpreter/prefix, platform, installed package
versions, import paths, resolved configuration, tensor runtime and regression
status. `status=pass` means import/runtime checks and pytest passed in that
setup attempt, **not** that exploration learned or reproduced a paper.
`status=imports_only` means `--skip-tests` was explicitly used and does not
qualify as a fully verified research environment.

Before a new attempt, an old report is renamed to
`environment_check.previous.<timestamp>.<PID>.json`. An installation failure
cannot leave an old `pass` at the current report path. A test failure leaves
the current report at `imports_passed_tests_pending`, returns nonzero and
prints the failing stage plus log path. Fix the reported cause and rerun the
same setup command; dependencies already installed can be reused.

Only one setup per profile may run at a time. A hard kill can leave the empty
`.setup/hpc.lock` or `.setup/local.lock` directory. Confirm no setup process
remains before removing that **specific empty lock** with `rmdir` and retrying.

To replace a damaged environment:

```bash
source ./setup.sh --hpc --rebuild
```

The previous directory is moved to `.conda-env.backup.<timestamp>.<PID>` (or
`.venv.backup...`), not deleted. Environment scripts contain absolute paths, so
restore the backup to its original path to use it again; do not activate it
from the backup name. Setup refuses to change a symlinked environment. It never
deletes collected data, reports or checkpoints. Backups consume disk space;
inspect and archive them before separately authorizing cleanup.

## Optional GPU setup and research gate

`source ./setup.sh --hpc --cuda` selects the pinned CUDA 12.1 wheels instead of
CPU-only wheels. It does not provide a GPU, driver or Slurm GPU allocation.
Default verification executes CPU tensors; CUDA training still needs a
separate hardware test. Switching an existing CUDA installation to CPU can
leave unused NVIDIA packages installed; setup does not purge unrelated or
orphaned packages automatically.

Once setup passes, run the [controlled MAPPO learning check](mappo/README.md)
and then review a 102,400-transition paper-map pilot before full research
training. Setup tests are not a substitute for those experiments.

## References and implementation choices

- The exact 2.3.1/0.18.1 versions and separate CPU/CUDA indexes follow
  [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/).
- Mandatory editable installs use [pip's local-project installation interface](https://pip.pypa.io/en/stable/topics/local-project-installs/).
- Bootstrap installer names and checksums come from the
  [official Miniforge 24.9.2-0 release](https://github.com/conda-forge/miniforge/releases/tag/24.9.2-0).
  Digests for each supported platform are pinned in `setup.sh`; downloads must
  match before an installer executes.

Regression tests in `tests/test_setup.py` simulate fresh/reused environments,
poisoned module variables, installation and test failures, safe rebuilding,
source activation and paths containing spaces without making network calls.
These test control flow, not Ada hardware or the remote availability of every
package archive.

## Recorded installer verification (2026-10-21 local / 2026-10-22 UTC)

- Created an isolated temporary checkout and an empty Python 3.8.20 venv,
  without reusing installed development packages; downloaded the complete stack.
- Ran the final complete installer successfully, including isolated imports,
  native operations, CLI launch outside the checkout and **121 passing tests**.
  Its regression phase took 26.89 seconds on the local macOS arm64 machine;
  this is not an Ada runtime estimate. Three legacy dependency deprecations
  were reported, without failing tests.
- Verified mandatory `onpolicy` and `explore_bench` editable imports resolve
  to the temporary checkout rather than the working development checkout.
- Saved a `status=pass` environment report, including the explicit known macOS
  grpcio metadata warning, at
  `/tmp/explore-setup-fresh.3mf727/.setup/environment_check.json`. This temporary
  development artifact is not permanent HPC research storage.
- Checked official Linux CPython 3.8 CPU-wheel resolution with pip's foreign-
  platform dry run: `torch==2.3.1+cpu` and `torchvision==0.18.1+cpu` are available.
  Those Linux binaries were not executed on the macOS host.
- No-provider Conda bootstrap, checksum failure, retries and rebuild safety
  were exercised with isolated test doubles. A real zero-provider Miniforge
  installation, actual Ada execution and CUDA hardware remain unverified here.

An early verification archive lacked `.git` and failed the research provenance
tests; setup now detects missing Git history before installing packages. The
successful run used a temporary Git snapshot. Neither verification attempt
changed the real checkout's Git history or its preexisting Python environment.
