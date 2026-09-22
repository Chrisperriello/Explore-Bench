#!/usr/bin/env bash

# Source this file so the virtual environment remains active in your shell:
#   source ./setup_env.sh
#
# Recreate it from scratch:
#   source ./setup_env.sh --rebuild
#
# Also install the MAPPO/onpolicy dependencies:
#   source ./setup_env.sh --full

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this script with: source ./setup_env.sh"
    exit 1
fi

explore_bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
explore_bench_venv="${explore_bench_root}/.venv"
explore_bench_rebuild=false
explore_bench_full=false

for explore_bench_arg in "$@"; do
    case "${explore_bench_arg}" in
        --rebuild)
            explore_bench_rebuild=true
            ;;
        --full)
            explore_bench_full=true
            ;;
        *)
            echo "Unknown option: ${explore_bench_arg}"
            echo "Usage: source ./setup_env.sh [--rebuild] [--full]"
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

python -m pip install --upgrade pip setuptools wheel || return 1
python -m pip install -r "${explore_bench_root}/requirements-grid.txt" || return 1

if [[ "${explore_bench_full}" == true ]]; then
    echo "Installing MAPPO/onpolicy dependencies..."
    python -m pip install torch torchvision || return 1
    python -m pip install -r "${explore_bench_root}/onpolicy/requirements.txt" || return 1
    python -m pip install -e "${explore_bench_root}/onpolicy" || return 1
fi

echo "Explore-Bench environment active: ${VIRTUAL_ENV}"
python --version
