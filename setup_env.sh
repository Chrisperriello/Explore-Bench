#!/usr/bin/env bash

# Backward-compatible setup + activation entry point. Full dependencies and
# editable packages are now mandatory; --full/--editable remain accepted.

explore_bench_activate() {
    local explore_activate_root explore_activate_profile explore_activate_arg explore_activate_prefix
    explore_activate_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    explore_activate_profile=local
    if [[ "$(uname -s)" == Linux ]]; then explore_activate_profile=hpc; fi
    for explore_activate_arg in "$@"; do
        case "$explore_activate_arg" in
            --hpc) explore_activate_profile=hpc ;;
            --local) explore_activate_profile=local ;;
            --help|-h) bash "${explore_activate_root}/setup.sh" --help; return ;;
        esac
    done
    bash "${explore_activate_root}/setup.sh" "$@" || return $?
    unset PYTHONHOME PYTHONPATH
    export PYTHONNOUSERSITE=1
    if [[ "$explore_activate_profile" == hpc ]]; then
        explore_activate_prefix="${explore_activate_root}/.conda-env"
        export MPLBACKEND=Agg PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    else
        explore_activate_prefix="${explore_activate_root}/.venv"
    fi
    if [[ -f "${explore_activate_prefix}/pyvenv.cfg" && -f "${explore_activate_prefix}/bin/activate" ]]; then
        source "${explore_activate_prefix}/bin/activate"
    else
        # Conda activation is not required for absolute interpreter paths.
        export PATH="${explore_activate_prefix}/bin:${PATH}"
        hash -r
    fi
    printf 'Explore-Bench interpreter active: %s\n' "${explore_activate_prefix}/bin/python"
}

explore_bench_activate "$@"
