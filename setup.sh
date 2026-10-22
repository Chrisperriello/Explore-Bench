#!/usr/bin/env bash
# Complete Level-0 setup. Run with bash; sourcing is handled by setup_env.sh.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/setup_env.sh" "$@"
    return $?
fi
set -Eeuo pipefail

explore_setup_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
explore_setup_system="$(uname -s)"
explore_setup_arch="$(uname -m)"
explore_setup_profile=local
if [[ "$explore_setup_system" == Linux ]]; then explore_setup_profile=hpc; fi
explore_setup_build=cpu
explore_setup_rebuild=false
explore_setup_tests=true
explore_setup_python_source=""

usage() {
    printf '%s\n' \
        'Usage: bash setup.sh [--hpc | --local] [--cpu | --cuda] [--rebuild]' \
        '                     [--python /path/to/python3.8] [--skip-tests]' \
        'Linux defaults to .conda-env; macOS defaults to .venv.' \
        'Installs the full Level-0/MAPPO stack and both local packages.' \
        'Dependency checks, outside-checkout imports and pytest run by default.' \
        '--rebuild preserves the old environment under a timestamped backup name.' \
        '--full and --editable are accepted for compatibility; both are now defaults.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hpc) explore_setup_profile=hpc ;;
        --local) explore_setup_profile=local ;;
        --cpu) explore_setup_build=cpu ;;
        --cuda) explore_setup_build=cu121 ;;
        --rebuild) explore_setup_rebuild=true ;;
        --skip-tests) explore_setup_tests=false ;;
        --full|--editable) ;; # No longer optional.
        --python)
            if [[ $# -lt 2 || "$2" != /* ]]; then usage >&2; exit 2; fi
            explore_setup_python_source="$2"
            shift
            ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "${explore_setup_system}/${explore_setup_arch}" in
    Linux/x86_64|Darwin/x86_64|Darwin/arm64) ;;
    *) printf 'Unsupported platform for the pinned PyTorch 2.3.1 setup: %s/%s\n' \
        "$explore_setup_system" "$explore_setup_arch" >&2; exit 2 ;;
esac
if [[ "$explore_setup_build" == cu121 && "$explore_setup_system" != Linux ]]; then
    echo '--cuda is supported only on Linux x86_64; use --cpu on macOS.' >&2
    exit 2
fi
if [[ "$explore_setup_profile" == hpc && -n "$explore_setup_python_source" ]]; then
    echo '--python is for --local venv creation; --hpc provisions Python with Conda.' >&2
    exit 2
fi
if ! command -v git >/dev/null 2>&1 || \
    ! git -C "$explore_setup_root" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo 'Run setup in a Git clone with a HEAD commit; research provenance requires Git history.' >&2
    exit 2
fi
if [[ "$explore_setup_profile" == hpc && "$explore_setup_tests" == true ]] && \
    command -v srun >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Run setup inside a compute allocation so its tests do not run on the login node:' >&2
    echo 'srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=02:00:00 --pty bash -l' >&2
    exit 2
fi

# Anaconda modules on Ada can set PYTHONHOME to a different Python installation.
# Do not depend on shell activation, inherited PYTHONPATH or user-site packages.
unset PYTHONHOME PYTHONPATH PIP_USER PIP_TARGET PIP_PREFIX PIP_REQUIRE_VIRTUALENV
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
explore_setup_state="${explore_setup_root}/.setup"
mkdir -p "$explore_setup_state"
explore_setup_lock="${explore_setup_state}/${explore_setup_profile}.lock"
if ! mkdir "$explore_setup_lock" 2>/dev/null; then
    printf 'Another setup may be running. Inspect the lock before retrying: %s\n' "$explore_setup_lock" >&2
    exit 1
fi
trap 'rmdir "$explore_setup_lock"' EXIT
explore_setup_log="${explore_setup_state}/setup_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee "$explore_setup_log") 2>&1
explore_setup_check="${explore_setup_state}/environment_check.json"
if [[ -f "$explore_setup_check" ]]; then
    mv -- "$explore_setup_check" "${explore_setup_state}/environment_check.previous.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
fi
explore_setup_stage=initialization
trap 'printf "Setup failed during %s (line %s). Log: %s\n" "$explore_setup_stage" "$LINENO" "$explore_setup_log" >&2' ERR
export MPLCONFIGDIR="${explore_setup_state}/matplotlib"
if [[ "$explore_setup_profile" == hpc ]]; then
    explore_setup_prefix="${explore_setup_root}/.conda-env"
else
    explore_setup_prefix="${explore_setup_root}/.venv"
fi
explore_setup_python="${explore_setup_prefix}/bin/python"
printf 'Setting up %s (%s); log: %s\n' "$explore_setup_prefix" "$explore_setup_build" "$explore_setup_log"
if [[ -L "$explore_setup_prefix" ]]; then
    echo 'Refusing to modify a symlinked environment. Use a repository-local environment directory.' >&2
    exit 1
fi
if [[ "$explore_setup_rebuild" == true && -e "$explore_setup_prefix" ]]; then
    explore_setup_backup="${explore_setup_prefix}.backup.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv -- "$explore_setup_prefix" "$explore_setup_backup"
    printf 'Preserved the previous environment at %s\n' "$explore_setup_backup"
fi

download() {
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --show-error --retry 3 --connect-timeout 30 \
            --output "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=60 --output-document="$2" "$1"
    else
        echo 'curl or wget is required to download the verified Python bootstrap.' >&2
        return 1
    fi
}

ensure_conda() {
    if command -v conda >/dev/null 2>&1; then return; fi
    # A non-login bash may not yet have the module function available.
    # Cluster-owned module scripts are not necessarily nounset-safe.
    set +u
    local explore_module_init
    for explore_module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
        if ! type module >/dev/null 2>&1 && [[ -f "$explore_module_init" ]]; then
            source "$explore_module_init"
        fi
    done
    if type module >/dev/null 2>&1; then
        module load python/anaconda3 || echo 'Ada module unavailable; using a private Miniforge bootstrap.'
        unset PYTHONHOME PYTHONPATH
    fi
    set -u
    if command -v conda >/dev/null 2>&1; then return; fi
    explore_setup_conda="${explore_setup_state}/miniforge/bin/conda"
    if [[ -x "$explore_setup_conda" ]]; then return; fi
    local explore_installer_platform explore_installer_hash explore_installer explore_release
    case "${explore_setup_system}/${explore_setup_arch}" in
        Linux/x86_64)
            explore_installer_platform=Linux-x86_64
            explore_installer_hash=ca8c544254c40ae5192eb7db4e133ff4eb9f942a1fec737dba8205ac3f626322 ;;
        Darwin/arm64)
            explore_installer_platform=MacOSX-arm64
            explore_installer_hash=b2ee4415bce51895b74b99de6e83ac0f737fd968052b5b49f79e05b292220084 ;;
        Darwin/x86_64)
            explore_installer_platform=MacOSX-x86_64
            explore_installer_hash=22b9eab7508d464c7d270883e733496beef25520d96c8383b4683f0122ae0222 ;;
    esac
    explore_release=Miniforge3-24.9.2-0-${explore_installer_platform}.sh
    explore_installer="${explore_setup_state}/${explore_release}"
    download "https://github.com/conda-forge/miniforge/releases/download/24.9.2-0/${explore_release}" "$explore_installer"
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s  %s\n' "$explore_installer_hash" "$explore_installer" | sha256sum --check -
    else
        printf '%s  %s\n' "$explore_installer_hash" "$explore_installer" | shasum -a 256 --check -
    fi
    # Do not pipe a remote script into a shell or change the user's shell startup.
    bash "$explore_installer" -b -p "${explore_setup_state}/miniforge"
}

explore_setup_stage='Python provisioning'
if [[ ! -e "$explore_setup_prefix" ]]; then
    if [[ "$explore_setup_profile" == local && -n "$explore_setup_python_source" ]]; then
        "$explore_setup_python_source" -I -c 'import sys; assert sys.version_info[:3] == (3, 8, 20), sys.version'
        "$explore_setup_python_source" -I -m venv "$explore_setup_prefix"
    elif [[ "$explore_setup_profile" == local ]] && command -v uv >/dev/null 2>&1; then
        uv venv --python 3.8.20 --seed "$explore_setup_prefix"
    elif [[ "$explore_setup_profile" == local ]] && command -v python3.8 >/dev/null 2>&1 && \
        python3.8 -I -c 'import sys; assert sys.version_info[:3] == (3, 8, 20)' >/dev/null 2>&1; then
        python3.8 -I -m venv "$explore_setup_prefix"
    else
        explore_setup_conda=conda
        ensure_conda
        # Keep bootstrap state local rather than adding channels to ~/.condarc.
        "$explore_setup_conda" create --yes --prefix "$explore_setup_prefix" \
            --override-channels --channel conda-forge python=3.8.20 pip=25.0.1
    fi
fi
if [[ ! -x "$explore_setup_python" ]]; then
    echo 'The existing environment is incomplete. Rerun with --rebuild to preserve and replace it.' >&2
    exit 1
fi
"$explore_setup_python" -I -c 'import pathlib, sys
assert sys.version_info[:3] == (3, 8, 20), "Python 3.8.20 required; use --rebuild: " + sys.version
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(), "Wrong environment prefix"
import encodings
print("Python:", sys.version.split()[0], "prefix:", sys.prefix)' "$explore_setup_prefix"

explore_setup_stage='packaging tools'
"$explore_setup_python" -m pip install --upgrade --no-input \
    pip==25.0.1 setuptools==75.3.4 wheel==0.45.1

explore_setup_stage='PyTorch runtime'
if [[ "$explore_setup_system" == Linux ]]; then
    echo "Installing the explicit ${explore_setup_build} PyTorch build (not the PyPI CUDA default)."
    # Select only these two wheel builds here. The complete requirements pass
    # below resolves their dependencies from PyPI, including pinned NumPy.
    "$explore_setup_python" -m pip install --no-input --no-deps \
        --index-url "https://download.pytorch.org/whl/${explore_setup_build}" \
        "torch==2.3.1+${explore_setup_build}" "torchvision==0.18.1+${explore_setup_build}"
else
    "$explore_setup_python" -m pip install --no-input --no-deps \
        torch==2.3.1 torchvision==0.18.1
fi

explore_setup_stage='Level-0 dependencies'
# Only one distribution may own cv2. Headless wheels avoid system X11/libGL
# dependencies on compute nodes; these are packages in this environment only.
explore_setup_conflicting_cv=$("$explore_setup_python" -I -c 'from importlib.metadata import distributions
names = {d.metadata["Name"].lower() for d in distributions()}
print(" ".join(sorted(names & {"opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"})))')
if [[ -n "$explore_setup_conflicting_cv" ]]; then
    read -r -a explore_setup_cv_packages <<< "$explore_setup_conflicting_cv"
    "$explore_setup_python" -m pip uninstall --yes "${explore_setup_cv_packages[@]}"
fi
"$explore_setup_python" -m pip install --no-input -r "${explore_setup_root}/requirements-level0-mappo.txt"

explore_setup_stage='editable project packages'
"$explore_setup_python" -m pip install --no-input --no-deps --no-build-isolation \
    --editable "${explore_setup_root}/onpolicy" --editable "$explore_setup_root"

explore_setup_stage='dependency and import checks'
"$explore_setup_python" -I "${explore_setup_root}/scripts/check_setup.py" --dependencies
explore_setup_expected_build="$explore_setup_build"
if [[ "$explore_setup_system" == Darwin ]]; then explore_setup_expected_build=native; fi
# -I deliberately removes current-directory/PYTHONPATH/user-site assistance.
"$explore_setup_python" -I "${explore_setup_root}/scripts/check_setup.py" \
    --root "$explore_setup_root" --prefix "$explore_setup_prefix" \
    --torch-build "$explore_setup_expected_build" --output "$explore_setup_check"

if [[ "$explore_setup_tests" == true ]]; then
    explore_setup_stage='regression suite'
    cd "$explore_setup_root"
    "$explore_setup_python" -m pytest -v --durations=10 -o faulthandler_timeout=120
fi
explore_setup_stage=complete
"$explore_setup_python" -I "${explore_setup_root}/scripts/check_setup.py" \
    --finish "$explore_setup_check" --tests "$explore_setup_tests" --log "$explore_setup_log"
printf '\nSetup complete. Interpreter: %s\nReport: %s\n' "$explore_setup_python" "$explore_setup_check"
printf 'No manual PYTHONPATH or editable-install step is needed.\n'
if [[ "$explore_setup_tests" != true ]]; then
    echo 'Regression tests were skipped; this is not a verified research environment.'
fi
