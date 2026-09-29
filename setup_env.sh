#!/usr/bin/env bash

# Source this file so the virtual environment remains active in your shell:
#   source ./setup_env.sh
#
# Recreate it from scratch:
#   source ./setup_env.sh --rebuild
#
# Also install the Level-0 MAPPO dependencies:
#   source ./setup_env.sh --full
#
# Install this checkout's onpolicy package in editable mode:
#   source ./setup_env.sh --full --editable

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this script with: source ./setup_env.sh"
    exit 1
fi

explore_bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
explore_bench_venv="${explore_bench_root}/.venv"
explore_bench_rebuild=false
explore_bench_full=false
explore_bench_editable=false

for explore_bench_arg in "$@"; do
    case "${explore_bench_arg}" in
        --rebuild)
            explore_bench_rebuild=true
            ;;
        --full)
            explore_bench_full=true
            ;;
        --editable)
            explore_bench_editable=true
            ;;
        *)
            echo "Unknown option: ${explore_bench_arg}"
            echo "Usage: source ./setup_env.sh [--rebuild] [--full] [--editable]"
            return 2
            ;;
    esac
done

if [[ "${explore_bench_rebuild}" == true && -d "${explore_bench_venv}" ]]; then
    if declare -F deactivate >/dev/null 2>&1; then
        deactivate
    fi
    rm -rf -- "${explore_bench_venv}"
fi

if [[ ! -d "${explore_bench_venv}" ]]; then
    echo "Creating Explore-Bench virtual environment..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.8 --seed "${explore_bench_venv}" || return 1
    elif command -v python3.8 >/dev/null 2>&1; then
        python3.8 -m venv "${explore_bench_venv}" || return 1
    else
        echo "Python 3.8 is required. Install it or install uv, then try again."
        return 1
    fi
fi

# shellcheck disable=SC1091
source "${explore_bench_venv}/bin/activate" || return 1

python -m pip install --upgrade \
    "pip<25.1" \
    "setuptools<76" \
    "wheel<0.46" || return 1
python -m pip install -r "${explore_bench_root}/requirements-grid.txt" || return 1

if [[ "${explore_bench_full}" == true ]]; then
    echo "Installing Level-0 MAPPO dependencies..."
    python -m pip install \
        -r "${explore_bench_root}/requirements-level0-mappo.txt" || return 1
fi

if [[ "${explore_bench_editable}" == true ]]; then
    echo "Installing the local onpolicy package in editable mode..."
    python -m pip install -e "${explore_bench_root}/onpolicy" || return 1
fi

python -m pip check || return 1

echo "Explore-Bench environment active: ${VIRTUAL_ENV}"
python --version
