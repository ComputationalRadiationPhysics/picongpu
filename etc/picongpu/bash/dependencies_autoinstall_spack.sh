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
# Dependency installer for the `bash/spack` preset.
#
# This script installs the same dependency set as dependencies_autoinstall.sh,
# but uses Spack into the per-package prefixes exported by the sourced profile.
# It is driven by `pic-deps install` / `pic-deps check`, or run directly:
#
#     PIC_PROFILE=... PIC_LIBS=$HOME/pic-libs/spack \
#         bash etc/picongpu/bash/dependencies_autoinstall_spack.sh [--check|--dry-run]
#
# Every version and every *_ROOT path is read from the profile; this script
# deliberately does not duplicate pins.

set -o pipefail

usage() {
    cat <<'USAGE'
Usage: dependencies_autoinstall_spack.sh [--check|--dry-run]

Modes:
  (default)       install every missing dependency into the profile's *_ROOT
  --check         report presence, change nothing, exit 1 on gaps
  --dry-run       print the install plan, change nothing
  -h, --help      show this help

Environment:
  PIC_PROFILE    required; profile sourced before anything else
  PIC_LIBS       required; writable install prefix (strict)
  Spack must be on PATH (source share/spack/setup-env.sh first).
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

log() { printf '[pic-deps-spack] %s\n' "$*"; }
die() {
    printf '[pic-deps-spack] ERROR: %s\n' "$*" >&2
    exit 1
}

# Versions and roots must come from the sourced profile, not from this script.
for _var in \
    BOOST_VERSION BLOSC_VERSION LIBPNG_VERSION PNGWRITER_VERSION \
    HDF5_VERSION ADIOS2_VERSION OPENPMD_VERSION FFTW_VERSION \
    BOOST_ROOT BLOSC_ROOT LIBPNG_ROOT PNGwriter_ROOT \
    HDF5_ROOT ADIOS2_ROOT OPENPMD_ROOT FFTW_ROOT; do
    if [ -z "${!_var:-}" ]; then
        die "the sourced profile does not export $_var"
    fi
done
unset _var

spack_install() { # SPEC ROOT
    local spec="$1" root="$2"
    if [ "$MODE" = "dry-run" ]; then
        printf '  [install] spack install --prefix=%s %s\n' "$root" "$spec"
        return 0
    fi
    log "spack install --prefix=$root $spec"
    spack install --prefix="$root" "$spec"
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
    printf 'Dependency status (PIC_LIBS=%s)\n' "$PIC_LIBS"
    for root in BOOST_ROOT BLOSC_ROOT LIBPNG_ROOT PNGwriter_ROOT HDF5_ROOT ADIOS2_ROOT OPENPMD_ROOT FFTW_ROOT; do
        show_status "$root" || rc=1
    done
    if [ "$rc" -eq 0 ]; then
        printf '\nAll dependencies are satisfied.\n'
    else
        printf "\nSome dependencies are missing; run without --check (or run pic-deps install).\n"
    fi
    return "$rc"
}

do_dry_run() {
    printf 'Install plan (PIC_LIBS=%s)\n' "$PIC_LIBS"
    if [ ! -d "$BOOST_ROOT" ]; then spack_install "boost@$BOOST_VERSION +program_options" "$BOOST_ROOT"; else printf '  [skip]    %-14s %s\n' BOOST_ROOT "$BOOST_ROOT"; fi
    if [ ! -d "$BLOSC_ROOT" ]; then spack_install "c-blosc2@$BLOSC_VERSION" "$BLOSC_ROOT"; else printf '  [skip]    %-14s %s\n' BLOSC_ROOT "$BLOSC_ROOT"; fi
    if [ ! -d "$LIBPNG_ROOT" ]; then spack_install "libpng@$LIBPNG_VERSION" "$LIBPNG_ROOT"; else printf '  [skip]    %-14s %s\n' LIBPNG_ROOT "$LIBPNG_ROOT"; fi
    if [ ! -d "$PNGwriter_ROOT" ]; then spack_install "pngwriter@$PNGWRITER_VERSION ^libpng" "$PNGwriter_ROOT"; else printf '  [skip]    %-14s %s\n' PNGwriter_ROOT "$PNGwriter_ROOT"; fi
    if [ ! -d "$HDF5_ROOT" ]; then spack_install "hdf5@$HDF5_VERSION +mpi +cxx" "$HDF5_ROOT"; else printf '  [skip]    %-14s %s\n' HDF5_ROOT "$HDF5_ROOT"; fi
    if [ ! -d "$ADIOS2_ROOT" ]; then spack_install "adios2@$ADIOS2_VERSION +mpi +hdf5 +blosc" "$ADIOS2_ROOT"; else printf '  [skip]    %-14s %s\n' ADIOS2_ROOT "$ADIOS2_ROOT"; fi
    if [ ! -d "$OPENPMD_ROOT" ]; then spack_install "openpmd-api@$OPENPMD_VERSION +mpi +hdf5 +adios2" "$OPENPMD_ROOT"; else printf '  [skip]    %-14s %s\n' OPENPMD_ROOT "$OPENPMD_ROOT"; fi
    if [ ! -d "$FFTW_ROOT" ]; then spack_install "fftw@$FFTW_VERSION" "$FFTW_ROOT"; else printf '  [skip]    %-14s %s\n' FFTW_ROOT "$FFTW_ROOT"; fi
    printf '\nNo changes made (--dry-run).\n'
}

do_install() {
    command -v spack >/dev/null 2>&1 || die "spack is not on PATH; source share/spack/setup-env.sh first"
    mkdir -p "$PIC_LIBS"

    if [ ! -d "$BOOST_ROOT" ]; then
        spack_install "boost@$BOOST_VERSION +program_options" "$BOOST_ROOT"
    else
        log "Boost already present at $BOOST_ROOT (skipping)"
    fi
    if [ ! -d "$BLOSC_ROOT" ]; then
        spack_install "c-blosc2@$BLOSC_VERSION" "$BLOSC_ROOT"
    else
        log "c-blosc2 already present at $BLOSC_ROOT (skipping)"
    fi
    if [ ! -d "$LIBPNG_ROOT" ]; then
        spack_install "libpng@$LIBPNG_VERSION" "$LIBPNG_ROOT"
    else
        log "libpng already present at $LIBPNG_ROOT (skipping)"
    fi
    if [ ! -d "$PNGwriter_ROOT" ]; then
        spack_install "pngwriter@$PNGWRITER_VERSION ^libpng" "$PNGwriter_ROOT"
    else
        log "PNGwriter already present at $PNGwriter_ROOT (skipping)"
    fi
    if [ ! -d "$HDF5_ROOT" ]; then
        spack_install "hdf5@$HDF5_VERSION +mpi +cxx" "$HDF5_ROOT"
    else
        log "HDF5 already present at $HDF5_ROOT (skipping)"
    fi
    if [ ! -d "$ADIOS2_ROOT" ]; then
        spack_install "adios2@$ADIOS2_VERSION +mpi +hdf5 +blosc" "$ADIOS2_ROOT"
    else
        log "ADIOS2 already present at $ADIOS2_ROOT (skipping)"
    fi
    if [ ! -d "$OPENPMD_ROOT" ]; then
        spack_install "openpmd-api@$OPENPMD_VERSION +mpi +hdf5 +adios2" "$OPENPMD_ROOT"
    else
        log "openPMD-api already present at $OPENPMD_ROOT (skipping)"
    fi
    if [ ! -d "$FFTW_ROOT" ]; then
        spack_install "fftw@$FFTW_VERSION" "$FFTW_ROOT"
    else
        log "FFTW already present at $FFTW_ROOT (skipping)"
    fi

    printf '\n'
    do_check
}

case "$MODE" in
    check) do_check ;;
    dry-run) do_dry_run ;;
    install) do_install ;;
esac
