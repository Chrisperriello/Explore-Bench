"""Exercise setup control flow without network, global installs or real Conda."""

import json
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]

# This stub records calls, models installed packages and simulates verifier /
# pytest results. It never imports the tested environment or downloads wheels.
FAKE_PYTHON = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["EXPLORE_SETUP_TEST_CALLS"], "a") as handle:
    handle.write(json.dumps({"args": args, "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonpath": os.environ.get("PYTHONPATH")}) + "\n")
if "-c" in args:
    if "from importlib.metadata" in args[args.index("-c") + 1]:
        print(os.environ.get("EXPLORE_SETUP_TEST_CV", ""))
elif "-m" in args:
    module = args[args.index("-m") + 1]
    if module == "pytest":
        sys.exit(int(os.environ.get("EXPLORE_SETUP_TEST_PYTEST_EXIT", "0")))
    if module == "pip" and "install" in args:
        sys.exit(int(os.environ.get("EXPLORE_SETUP_TEST_INSTALL_EXIT", "0")))
elif any(arg.endswith("check_setup.py") for arg in args):
    if "--dependencies" in args:
        sys.exit(0)
    if "--finish" in args:
        path = pathlib.Path(args[args.index("--finish") + 1])
        record = json.loads(path.read_text())
        record["status"] = "pass" if args[args.index("--tests") + 1] == "true" else "imports_only"
    else:
        path = pathlib.Path(args[args.index("--output") + 1])
        record = {"status": "imports_passed_tests_pending"}
    path.write_text(json.dumps(record))
'''


def executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout with spaces"
    repo.mkdir()
    for name in ("setup.sh", "setup_env.sh", "requirements-grid.txt", "requirements-level0-mappo.txt"):
        shutil.copy2(ROOT / name, repo / name)
    tools = tmp_path / "bin"
    tools.mkdir()
    fake = tools / "fake_python"
    executable(fake, "#!{}\n{}".format(sys.executable, FAKE_PYTHON))
    executable(tools / "uname", '#!/bin/bash\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; esac\n')
    executable(tools / "git", '#!/bin/bash\nexit 0\n')
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
               SLURM_JOB_ID="123", PYTHONHOME="/broken/module/python", PYTHONPATH="/unrelated/tree",
               EXPLORE_SETUP_TEST_CALLS=str(tmp_path / "calls.jsonl"),
               EXPLORE_SETUP_TEST_PYTHON=str(fake))
    return repo, tools, fake, env


def existing_environment(checkout):
    repo, _, fake, _ = checkout
    binary = repo / ".conda-env/bin/python"
    binary.parent.mkdir(parents=True)
    shutil.copy2(fake, binary)
    return binary


def run_setup(checkout, *flags):
    repo, _, _, env = checkout
    return subprocess.run(["bash", str(repo / "setup.sh"), "--hpc", *flags],
                          cwd=repo.parent, env=env, text=True, capture_output=True, timeout=30)


def calls(checkout):
    path = Path(checkout[3]["EXPLORE_SETUP_TEST_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def fake_conda(checkout):
    _, tools, _, _ = checkout
    executable(tools / "conda", '#!/bin/bash\nset -e\n'
               'while [[ "$1" != --prefix ]]; do shift; done\nshift\n'
               'mkdir -p "$1/bin"\ncp "$EXPLORE_SETUP_TEST_PYTHON" "$1/bin/python"\n')


def test_existing_hpc_setup_clears_poison_and_installs_everything(checkout):
    existing_environment(checkout)
    checkout[3]["EXPLORE_SETUP_TEST_CV"] = "opencv-python"
    result = run_setup(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = calls(checkout)
    assert all(row["pythonhome"] is None and row["pythonpath"] is None for row in recorded)
    commands = [row["args"] for row in recorded]
    torch_install = next(args for args in commands if "torch==2.3.1+cpu" in args)
    assert "--no-deps" in torch_install
    assert torch_install[torch_install.index("--index-url") + 1] == "https://download.pytorch.org/whl/cpu"
    assert any("uninstall" in args and "opencv-python" in args for args in commands)
    packages = next(args for args in commands if "--editable" in args)
    assert str(checkout[0] / "onpolicy") in packages and str(checkout[0]) in packages
    assert any("--dependencies" in args for args in commands)
    assert any("-I" in args and "--torch-build" in args for args in commands)
    assert any("pytest" in args for args in commands)
    report = json.loads((checkout[0] / ".setup/environment_check.json").read_text())
    assert report["status"] == "pass"
    assert not (checkout[0] / ".setup/hpc.lock").exists()


def test_fresh_hpc_setup_provisions_local_python(checkout):
    fake_conda(checkout)
    result = run_setup(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (checkout[0] / ".conda-env/bin/python").is_file()
    assert not (checkout[0] / ".venv").exists()


def test_failed_tests_do_not_publish_a_success_report(checkout):
    existing_environment(checkout)
    checkout[3]["EXPLORE_SETUP_TEST_PYTEST_EXIT"] = "1"
    result = run_setup(checkout)
    assert result.returncode == 1
    assert "regression suite" in result.stdout + result.stderr
    assert not any("--finish" in row["args"] for row in calls(checkout))
    report = json.loads((checkout[0] / ".setup/environment_check.json").read_text())
    assert report["status"] == "imports_passed_tests_pending"
    assert not (checkout[0] / ".setup/hpc.lock").exists()


def test_failed_install_stops_and_can_be_retried(checkout):
    existing_environment(checkout)
    checkout[3]["EXPLORE_SETUP_TEST_INSTALL_EXIT"] = "1"
    first = run_setup(checkout)
    assert first.returncode == 1
    assert not any("pytest" in row["args"] for row in calls(checkout))
    assert not (checkout[0] / ".setup/environment_check.json").exists()
    checkout[3]["EXPLORE_SETUP_TEST_INSTALL_EXIT"] = "0"
    second = run_setup(checkout)
    assert second.returncode == 0, second.stdout + second.stderr


def test_rebuild_preserves_old_environment_and_previous_report(checkout):
    existing_environment(checkout)
    fake_conda(checkout)
    sentinel = checkout[0] / ".conda-env/keep.txt"
    sentinel.write_text("recoverable")
    state = checkout[0] / ".setup"
    state.mkdir()
    (state / "environment_check.json").write_text('{"status":"old"}')
    result = run_setup(checkout, "--rebuild")
    assert result.returncode == 0, result.stdout + result.stderr
    backups = list(checkout[0].glob(".conda-env.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "keep.txt").read_text() == "recoverable"
    previous = list(state.glob("environment_check.previous.*.json"))
    assert len(previous) == 1 and json.loads(previous[0].read_text())["status"] == "old"


def test_skip_tests_is_explicitly_not_a_verified_environment(checkout):
    existing_environment(checkout)
    result = run_setup(checkout, "--skip-tests", "--full", "--editable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any("pytest" in row["args"] for row in calls(checkout))
    report = json.loads((checkout[0] / ".setup/environment_check.json").read_text())
    assert report["status"] == "imports_only"


def test_sourced_setup_clears_the_calling_shell_and_activates_python(checkout):
    binary = existing_environment(checkout)
    # Conda's activate entry point must not accidentally activate its base env.
    (binary.parent / "activate").write_text("return 39\n")
    result = subprocess.run(
        ["bash", "-c", 'source "$1/setup.sh" --hpc || exit $?\n'
         '[[ -z "${PYTHONHOME:-}" && -z "${PYTHONPATH:-}" ]] || exit 9\n'
         '[[ "$MPLBACKEND" == Agg && "$OMP_NUM_THREADS" == 1 ]] || exit 10\n'
         'command -v python', "setup-test", str(checkout[0])],
        env=checkout[3], text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(binary) in result.stdout


def test_symlinked_environment_is_not_modified(checkout):
    repo, _, _, _ = checkout
    unrelated = repo.parent / "unrelated"
    unrelated.mkdir()
    (repo / ".conda-env").symlink_to(unrelated, target_is_directory=True)
    result = run_setup(checkout, "--rebuild")
    assert result.returncode != 0
    assert (repo / ".conda-env").is_symlink()
    assert list(unrelated.iterdir()) == []


def test_unknown_options_fail_before_installing(checkout):
    result = run_setup(checkout, "--not-a-real-option")
    assert result.returncode == 2
    assert calls(checkout) == []


def test_missing_git_history_fails_before_installing(checkout):
    executable(checkout[1] / "git", '#!/bin/bash\nexit 128\n')
    result = run_setup(checkout)
    assert result.returncode == 2
    assert "Git clone" in result.stdout + result.stderr
    assert calls(checkout) == []


def test_cuda_build_requires_explicit_selection(checkout):
    existing_environment(checkout)
    result = run_setup(checkout, "--cuda")
    assert result.returncode == 0, result.stdout + result.stderr
    torch_install = next(row["args"] for row in calls(checkout) if "torch==2.3.1+cu121" in row["args"])
    assert torch_install[torch_install.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu121"


def standalone_bootstrap_tools(checkout):
    repo, tools, _, env = checkout
    standalone = repo.parent / "standalone-bin"
    standalone.mkdir()
    # Deliberately exclude installed Python, Conda, uv, modules and Slurm.
    for name in ("bash", "dirname", "mkdir", "rmdir", "date", "tee", "mv", "cp"):
        (standalone / name).symlink_to(shutil.which(name))
    for name in ("sha256sum", "shasum"):
        if shutil.which(name):
            (standalone / name).symlink_to(shutil.which(name))
    shutil.copy2(tools / "uname", standalone / "uname")
    shutil.copy2(tools / "git", standalone / "git")
    env["PATH"] = str(standalone)
    installer = repo.parent / "installer-fixture.sh"
    env["EXPLORE_SETUP_TEST_INSTALLER"] = str(installer)
    executable(standalone / "curl", '#!/bin/bash\nset -e\n'
               'while [[ "$1" != --output ]]; do shift; done\nshift\n'
               'cp "$EXPLORE_SETUP_TEST_INSTALLER" "$1"\n')
    return standalone, installer


def run_without_module_or_provider(checkout):
    return subprocess.run(
        ["bash", "-c", 'module() { return 1; }\nexport -f module\nexec bash "$1" --hpc',
         "setup-bootstrap-test", str(checkout[0] / "setup.sh")],
        env=checkout[3], text=True, capture_output=True, timeout=30,
    )


def test_bootstrap_rejects_incorrect_checksum_before_executing(checkout):
    _, installer = standalone_bootstrap_tools(checkout)
    installer.write_text('#!/bin/bash\nexit 99\n')
    result = run_without_module_or_provider(checkout)
    assert result.returncode != 0
    assert "FAILED" in result.stdout + result.stderr
    assert not (checkout[0] / ".setup/miniforge").exists()
    assert calls(checkout) == []


def test_no_provider_bootstrap_creates_environment_after_checksum_step(checkout):
    fake_conda(checkout)
    standalone, installer = standalone_bootstrap_tools(checkout)
    # The installer and checksum tool are test doubles; no remote code executes.
    checkout[3]["EXPLORE_SETUP_TEST_CONDA"] = str(checkout[1] / "conda")
    installer.write_text('#!/bin/bash\nset -e\n'
                         'while [[ "$1" != -p ]]; do shift; done\nshift\n'
                         'mkdir -p "$1/bin"\ncp "$EXPLORE_SETUP_TEST_CONDA" "$1/bin/conda"\n')
    checksum_name = "sha256sum" if (standalone / "sha256sum").exists() else "shasum"
    (standalone / checksum_name).unlink()
    executable(standalone / checksum_name, '#!{}\n'.format(sys.executable) +
               'import sys\nvalue = sys.stdin.read()\n'
               'assert value.startswith("ca8c544254c40ae5192eb7db4e133ff4eb9f942a1fec737dba8205ac3f626322  ")\n')
    result = run_without_module_or_provider(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (checkout[0] / ".setup/miniforge/bin/conda").exists()
    assert (checkout[0] / ".conda-env/bin/python").exists()


@pytest.fixture
def dependency_checker():
    spec = importlib.util.spec_from_file_location("setup_checks", ROOT / "scripts/check_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_check_preserves_strict_failures(dependency_checker, monkeypatch):
    monkeypatch.setattr(dependency_checker.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="broken dependency conflict", stderr=""))
    with pytest.raises(RuntimeError, match="broken dependency conflict"):
        dependency_checker.check_dependencies()


def test_clean_dependency_check(dependency_checker, monkeypatch):
    monkeypatch.setattr(dependency_checker.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr=""))
    assert dependency_checker.check_dependencies()["status"] == "pass"


def test_known_grpc_metadata_warning_is_recorded_after_native_import(dependency_checker, monkeypatch):
    monkeypatch.setattr(dependency_checker.sys, "platform", "darwin")
    monkeypatch.setattr(dependency_checker.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(dependency_checker.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1,
                                        stdout="grpcio 1.70.0 is not supported on this platform\n", stderr=""))
    monkeypatch.setattr(dependency_checker, "distribution", lambda name:
                        SimpleNamespace(version="1.70.0", read_text=lambda name:
                                        "Tag: cp38-cp38-macosx_10_14_x86_64\n"))
    record = dependency_checker.check_dependencies()
    assert record["status"] == "known_metadata_warning"
    assert record["native_extension"].endswith(".so")


@pytest.mark.parametrize("platform_name,machine,output", [
    ("linux", "x86_64", "grpcio 1.70.0 is not supported on this platform"),
    ("darwin", "arm64", "grpcio 1.70.0 is not supported on this platform\nbroken dependency"),
    ("darwin", "arm64", "anotherpackage is not supported on this platform"),
])
def test_grpc_exception_does_not_hide_other_platform_or_dependency_errors(
        dependency_checker, monkeypatch, platform_name, machine, output):
    monkeypatch.setattr(dependency_checker.sys, "platform", platform_name)
    monkeypatch.setattr(dependency_checker.platform, "machine", lambda: machine)
    monkeypatch.setattr(dependency_checker.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout=output, stderr=""))
    with pytest.raises(RuntimeError, match="Dependency validation failed"):
        dependency_checker.check_dependencies()
