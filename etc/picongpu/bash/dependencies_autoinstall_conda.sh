#!/usr/bin/env bash
# shellcheck disable=SC2154 # *_ROOT is validated and read from the sourced profile
# Copyright 2026 PIConGPU contributors
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#
# Dependency installer for the `bash/conda` preset.
#
# This script creates one conda-forge environment at `$CONDA_PREFIX`
# (`$PIC_LIBS/conda`) that provides the seven library roots, and builds
# PNGwriter into its own per-package prefix because conda-forge does not ship
# it. It is driven by `pic-deps install` / `pic-deps check`, or run directly:
#
#     PIC_PROFILE=... PIC_LIBS=$HOME/pic-libs \
#         bash etc/picongpu/bash/dependencies_autoinstall_conda.sh [--check|--dry-run]
#
# Versions and roots are read from the sourced profile. conda-forge may not
# carry the exact pins, so they are used as lower-bound constraints and the
# resolved versions are reported; this script does not hard-fail on a mismatch.

set -o pipefail

usage() {
    cat <<'USAGE'
Usage: dependencies_autoinstall_conda.sh [--check|--dry-run]

Modes:
  (default)       create/complete the conda environment and build PNGwriter
  --check         report presence and resolved versions, exit 1 on gaps
  --dry-run       print the install plan, change nothing
  -h, --help      show this help

Environment:
  PIC_PROFILE         required; profile sourced before anything else
  PIC_LIBS            required; writable install prefix (strict)
  PIC_DEPS_CONDA      conda executable to use (default: micromamba, mamba, conda)
  PIC_DEPS_BUILD_DIR  scratch/build tree (default: $PIC_LIBS/.build)
USAGE
}

MODE="install"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) MODE="check" ;;
        --dry-run) MODE="dry-run" ;;
        -h|--help) usage; exit 0 ;;
        *) printf "error: unknown argument '%s' (try --help)\n" "$1" >&2; exit 2 ;;
    esac
    shift
done

if [ ! -f "${PIC_PROFILE:-}" ]; then
    printf "Source a profile!\n" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$PIC_PROFILE"

set -euf -o pipefail

if [ -z "${PIC_LIBS:-}" ]; then
    printf "error: PIC_LIBS is not set.\n" >&2
    printf "       Set rc_params['pic_libs'] (or export PIC_LIBS) to a writable prefix.\n" >&2
    exit 2
fi

have() { command -v "$1" >/dev/null 2>&1; }
log() { printf '[pic-deps-conda] %s\n' "$*"; }
die() {
    printf '[pic-deps-conda] ERROR: %s\n' "$*" >&2
    exit 1
}

# Versions and roots must come from the sourced profile, not from this script.
for _var in \
    BOOST_VERSION BLOSC_VERSION LIBPNG_VERSION PNGWRITER_VERSION \
    HDF5_VERSION ADIOS2_VERSION OPENPMD_VERSION FFTW_VERSION \
    CONDA_PREFIX BOOST_ROOT BLOSC_ROOT LIBPNG_ROOT PNGwriter_ROOT \
    HDF5_ROOT ADIOS2_ROOT OPENPMD_ROOT FFTW_ROOT; do
    if [ -z "${!_var:-}" ]; then
        die "the sourced profile does not export $_var"
    fi
done
unset _var

SOURCE_DIR="${PIC_DEPS_BUILD_DIR:-$PIC_LIBS/.build}"
_env_planned=0

detect_conda() {
    if [ -n "${PIC_DEPS_CONDA:-}" ]; then
        CONDA_BIN="$PIC_DEPS_CONDA"
    elif have micromamba; then
        CONDA_BIN="micromamba"
    elif have mamba; then
        CONDA_BIN="mamba"
    elif have conda; then
        CONDA_BIN="conda"
    else
        die "neither micromamba nor conda is available; set PIC_DEPS_CONDA"
    fi
    command -v "$CONDA_BIN" >/dev/null 2>&1 || die "'$CONDA_BIN' is not executable"
}

run_in_env() {
    "$CONDA_BIN" run -p "$CONDA_PREFIX" "$@"
}

conda_specs() {
    # conda-forge constraints derived from the profile pins. Errors/absence of an
    # exact version are resolved by the solver and reported below.
    printf '%s\n' \
        'gxx_linux-64>=10' \
        'cmake>=3.28' \
        openmpi \
        "boost>=${BOOST_VERSION%.*}" \
        'hdf5=*=mpi*' \
        'adios2=*=mpi*' \
        "openpmd-api>=${OPENPMD_VERSION%.*}" \
        "fftw>=${FFTW_VERSION%.*}" \
        "c-blosc2>=${BLOSC_VERSION%.*}" \
        "libpng>=${LIBPNG_VERSION%.*}" \
        zlib pkg-config make git curl
}

create_env() {
    if [ -d "$CONDA_PREFIX/conda-meta" ]; then
        log "Conda environment already present at $CONDA_PREFIX (skipping create)"
        return 0
    fi
    if [ "$_env_planned" -eq 1 ]; then
        return 0
    fi
    if [ "$MODE" = "dry-run" ]; then
        printf '  [install] %s create -p %s -c conda-forge (conda-forge environment)\n' "$CONDA_BIN" "$CONDA_PREFIX"
        _env_planned=1
        return 0
    fi
    mkdir -p "$PIC_LIBS"
    log "creating conda-forge environment at $CONDA_PREFIX"
    local specs
    mapfile -t specs < <(conda_specs)
    "$CONDA_BIN" create -y -p "$CONDA_PREFIX" -c conda-forge --override-channels "${specs[@]}"
}

fetch() { # URL DEST
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    log "downloading $(basename "$dest")"
    if have curl; then
        curl -fL --retry 3 --connect-timeout 30 -o "$dest.part" "$url"
    elif have wget; then
        wget -O "$dest.part" "$url"
    else
        die "neither curl nor wget is available"
    fi
    mv "$dest.part" "$dest"
}

build_pngwriter() {
    if [ -d "$PNGwriter_ROOT" ]; then
        log "PNGwriter already present at $PNGwriter_ROOT (skipping)"
        return 0
    fi
    local archive="$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}.tar.gz"
    if [ "$MODE" = "dry-run" ]; then
        printf '  [install] build PNGwriter %s into %s\n' "$PNGWRITER_VERSION" "$PNGwriter_ROOT"
        return 0
    fi
    fetch "https://github.com/pngwriter/pngwriter/archive/refs/tags/${PNGWRITER_VERSION}.tar.gz" "$archive"
    mkdir -p "$SOURCE_DIR"
    tar -xzf "$archive" -C "$SOURCE_DIR"
    log "building PNGwriter $PNGWRITER_VERSION into $PNGwriter_ROOT"
    run_in_env cmake -S "$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}" -B "$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}-build" \
        -DCMAKE_INSTALL_PREFIX="$PNGwriter_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DPNGwriter_USE_FREETYPE=OFF \
        -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
    run_in_env cmake --build "$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}-build" -j "$(nproc 2>/dev/null || echo 4)" --target install
}

report_versions() {
    if [ -d "$CONDA_PREFIX/conda-meta" ] && [ -n "${CONDA_BIN:-}" ]; then
        printf 'Resolved conda-forge versions:\n'
        "$CONDA_BIN" list -p "$CONDA_PREFIX" 2>/dev/null \
            | awk '$1 ~ /^(boost|hdf5|adios2|openpmd-api|fftw|c-blosc2|libpng|zlib|openmpi)$/ {printf "  %-14s %s\n", $1, $2}' \
            || true
    fi
}

show_status() { # ROOT
    local root="$1"
    if [ -d "${!root}" ]; then
        printf '  [ok]      %-14s %s\n' "$root" "${!root}"
    else
        printf '  [MISSING] %-14s %s\n' "$root" "${!root}"
        return 1
    fi
}

do_check() {
    local rc=0 root
    printf 'Dependency status (PIC_LIBS=%s, CONDA_PREFIX=%s)\n' "$PIC_LIBS" "$CONDA_PREFIX"
    for root in BOOST_ROOT BLOSC_ROOT LIBPNG_ROOT PNGwriter_ROOT HDF5_ROOT ADIOS2_ROOT OPENPMD_ROOT FFTW_ROOT; do
        show_status "$root" || rc=1
    done
    report_versions
    if [ "$rc" -eq 0 ]; then
        printf '\nAll dependencies are satisfied.\n'
    else
        printf "\nSome dependencies are missing; run without --check (or run pic-deps install).\n"
    fi
    return "$rc"
}

do_dry_run() {
    detect_conda
    printf 'Install plan (PIC_LIBS=%s, CONDA_PREFIX=%s)\n' "$PIC_LIBS" "$CONDA_PREFIX"
    create_env
    if [ ! -d "$BOOST_ROOT" ]; then create_env; fi
    if [ ! -d "$BLOSC_ROOT" ]; then create_env; fi
    if [ ! -d "$LIBPNG_ROOT" ]; then create_env; fi
    if [ ! -d "$PNGwriter_ROOT" ]; then build_pngwriter; fi
    if [ ! -d "$HDF5_ROOT" ]; then create_env; fi
    if [ ! -d "$ADIOS2_ROOT" ]; then create_env; fi
    if [ ! -d "$OPENPMD_ROOT" ]; then create_env; fi
    if [ ! -d "$FFTW_ROOT" ]; then create_env; fi
    printf '\nNo changes made (--dry-run).\n'
}

do_install() {
    detect_conda
    mkdir -p "$PIC_LIBS"

    if [ ! -d "$BOOST_ROOT" ]; then
        create_env
    fi
    if [ ! -d "$BLOSC_ROOT" ]; then
        create_env
    fi
    if [ ! -d "$LIBPNG_ROOT" ]; then
        create_env
    fi
    if [ ! -d "$PNGwriter_ROOT" ]; then
        build_pngwriter
    else
        log "PNGwriter already present at $PNGwriter_ROOT (skipping)"
    fi
    if [ ! -d "$HDF5_ROOT" ]; then
        create_env
    fi
    if [ ! -d "$ADIOS2_ROOT" ]; then
        create_env
    fi
    if [ ! -d "$OPENPMD_ROOT" ]; then
        create_env
    fi
    if [ ! -d "$FFTW_ROOT" ]; then
        create_env
    fi

    printf '\n'
    do_check
}

case "$MODE" in
    check) do_check ;;
    dry-run) do_dry_run ;;
    install) do_install ;;
esac
