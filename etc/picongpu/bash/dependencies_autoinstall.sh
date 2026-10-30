#!/usr/bin/env bash
# shellcheck disable=SC2046 # package groups are intentionally word-split
# Copyright 2023-2026 Axel Huebl, Marco Garten, Klaus Steiniger, Pawel Ordyna,
#                     Richard Pausch, Julian Lenz
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
# setup for local Linux workstations via the "bash" preset
# last updated: 2026-10-30
#
# This is the module-less counterpart of the cluster dependency scripts. It
# installs PIConGPU's compile-time dependencies into the user-writable prefix
# `$PIC_LIBS` (the value of `rc_params['pic_libs']`, exported by the rendered
# profile) and is driven by
#
#     pic-deps install
#     pic-deps check
#
# or run directly:
#
#     PIC_LIBS=$HOME/pic-libs bash etc/picongpu/bash/dependencies_autoinstall.sh [--check|--dry-run]
#
# Options:
#   (default)       install every missing dependency into $PIC_LIBS
#   --check         report presence/versions, change nothing, exit 1 on gaps
#   --dry-run       print the install plan, change nothing
#   --only LIST     restrict install/dry-run to comma-separated dependency names
#                   (boost,blosc,libpng,pngwriter,hdf5,adios2,openpmd,fftw)
#                   also honoured via PIC_DEPS_ONLY
#   -h, --help      show this help
#
# Environment:
#   PIC_PROFILE            required; profile sourced before anything else
#   PIC_LIBS               required; writable install prefix (strict)
#   PIC_DEPS_BUILD_DIR     scratch/build tree (default: $PIC_LIBS/.build)
#   PIC_DEPS_JOBS          parallel build jobs (default: nproc)
#   *_VERSION              pins; the bash profile exports the same variables
#
# Policy: libraries are built from source into $PIC_LIBS so that the
# directory guards pic-deps discovers match the profile's `*_ROOT` exports.
# Distro packages are used only for the toolchain/basics
# (compiler, make, git/curl/tar/bzip2, pkg-config, zlib, MPI, and CMake when
# the packaged version meets the 3.28 floor). No environment modules are used.
# macOS is intentionally out of scope (Linux only, documented).
#
# Air-gapped/mirror overrides are deliberately deferred to a follow-up; every
# download below currently uses its upstream HTTPS/FTP URL.

set -o pipefail

usage() {
    cat <<'EOF'
Usage: dependencies_autoinstall.sh [--check|--dry-run] [--only LIST]

Modes:
  (default)       install every missing dependency into $PIC_LIBS
  --check         report presence/versions, change nothing, exit 1 on gaps
  --dry-run       print the install plan, change nothing
  --only LIST     restrict install/dry-run to comma-separated dependency names
                  (boost,blosc,libpng,pngwriter,hdf5,adios2,openpmd,fftw)
                  also honoured via PIC_DEPS_ONLY
  -h, --help      show this help

Environment:
  PIC_PROFILE            required; profile sourced before anything else
  PIC_LIBS               required; writable install prefix (strict)
  PIC_DEPS_BUILD_DIR     scratch/build tree (default: $PIC_LIBS/.build)
  PIC_DEPS_JOBS          parallel build jobs (default: nproc)

The script is normally driven by `pic-deps install` / `pic-deps check`, which
render and source the profile from rc_params before invoking it.
EOF
}

MODE="install"
ONLY="${PIC_DEPS_ONLY:-}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            MODE="check"
            ;;
        --dry-run)
            MODE="dry-run"
            ;;
        --only)
            shift
            [ "$#" -gt 0 ] || {
                printf "error: --only needs a comma-separated argument\n" >&2
                exit 2
            }
            ONLY="$1"
            ;;
        --only=*)
            ONLY="${1#--only=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf "error: unknown argument '%s' (try --help)\n" "$1" >&2
            exit 2
            ;;
    esac
    shift
done

# get PIConGPU profile
if [ ! -f "${PIC_PROFILE:-}" ]; then
    printf "Source a profile!\n"
    exit 1
else
    # shellcheck source=/dev/null
    source "$PIC_PROFILE"
fi

set -euf -o pipefail

# Strict prefix: pic-deps renders the profile from rc_params['pic_libs'], so an
# unset PIC_LIBS means the profile was not rendered/sourced as intended.
if [ -z "${PIC_LIBS:-}" ]; then
    printf "error: PIC_LIBS is not set.\n" >&2
    printf "       Set rc_params['pic_libs'] (or export PIC_LIBS) to a writable prefix.\n" >&2
    exit 2
fi

# Pins: the newest version pinned by any of the cluster dependency scripts.
# The bash profile exports the same variables so that its *_ROOT entries and
# this script cannot drift apart. All are overridable.
CMAKE_MIN_VERSION="${CMAKE_MIN_VERSION:-3.28}"
CMAKE_VERSION="${CMAKE_VERSION:-3.31.6}"
BOOST_VERSION="${BOOST_VERSION:-1.87.0}"
BLOSC_VERSION="${BLOSC_VERSION:-2.22.0}"
LIBPNG_VERSION="${LIBPNG_VERSION:-1.6.34}"
PNGWRITER_VERSION="${PNGWRITER_VERSION:-0.7.0}"
HDF5_VERSION="${HDF5_VERSION:-2.0.0}"
ADIOS2_VERSION="${ADIOS2_VERSION:-2.11.0}"
OPENPMD_VERSION="${OPENPMD_VERSION:-0.17.1}"
FFTW_VERSION="${FFTW_VERSION:-3.3.10}"

# Roots normally come from the sourced profile; the fallbacks keep a direct
# invocation working with a hand-written profile that does not export them.
BOOST_ROOT="${BOOST_ROOT:-$PIC_LIBS/boost/$BOOST_VERSION}"
BLOSC_ROOT="${BLOSC_ROOT:-$PIC_LIBS/c-blosc2/$BLOSC_VERSION}"
LIBPNG_ROOT="${LIBPNG_ROOT:-$PIC_LIBS/libpng/$LIBPNG_VERSION}"
PNGwriter_ROOT="${PNGwriter_ROOT:-$PIC_LIBS/pngwriter/$PNGWRITER_VERSION}"
HDF5_ROOT="${HDF5_ROOT:-$PIC_LIBS/hdf5/$HDF5_VERSION}"
ADIOS2_ROOT="${ADIOS2_ROOT:-$PIC_LIBS/adios2/$ADIOS2_VERSION}"
OPENPMD_ROOT="${OPENPMD_ROOT:-$PIC_LIBS/openpmd-api/$OPENPMD_VERSION}"
FFTW_ROOT="${FFTW_ROOT:-$PIC_LIBS/fftw/$FFTW_VERSION}"
CMAKE_ROOT="${CMAKE_ROOT:-$PIC_LIBS/cmake/$CMAKE_VERSION}"

SOURCE_DIR="${PIC_DEPS_BUILD_DIR:-$PIC_LIBS/.build}"
JOBS="${PIC_DEPS_JOBS:-$(nproc 2>/dev/null || echo 4)}"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
log() { printf '[pic-deps] %s\n' "$*"; }
warn() { printf '[pic-deps] WARNING: %s\n' "$*" >&2; }
die() {
    printf '[pic-deps] ERROR: %s\n' "$*" >&2
    exit 1
}
have() { command -v "$1" >/dev/null 2>&1; }

# version_ge ACTUAL REQUIRED (both dot-separated; empty ACTUAL is never >=)
version_ge() {
    [ -n "${1:-}" ] || return 1
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

is_enabled() {
    [ -z "$ONLY" ] || case ",$ONLY," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# toolchain detection (version-aware where a version matters)
# ---------------------------------------------------------------------------
compiler_version() {
    if have c++; then
        c++ -dumpfullversion -dumpversion 2>/dev/null | head -n1
    elif have g++; then
        g++ -dumpfullversion -dumpversion 2>/dev/null | head -n1
    elif have clang++; then
        clang++ --version 2>/dev/null | sed -n '1s/.*version \([0-9.]*\).*/\1/p'
    fi
}

compiler_ok() {
    have cc && have c++ || return 1
    local v
    v="$(compiler_version || true)"
    if have clang++ && ! have g++; then
        version_ge "$v" "11"
    else
        version_ge "$v" "10"
    fi
}

cmake_version() {
    have cmake || return 1
    cmake --version 2>/dev/null | sed -n '1s/^cmake version \([0-9.]*\).*/\1/p'
}

cmake_ok() { version_ge "$(cmake_version || true)" "$CMAKE_MIN_VERSION"; }

mpi_ok() { have mpicc && have mpicxx && have mpiexec; }

mpi_version() {
    if have mpiexec; then
        mpiexec --version 2>/dev/null | head -n1
    elif have mpicxx; then
        mpicxx --version 2>/dev/null | head -n1
    else
        return 1
    fi
}

zlib_ok() {
    if have pkg-config && pkg-config --exists zlib 2>/dev/null; then
        return 0
    fi
    [ -f /usr/include/zlib.h ] || [ -f /usr/include/x86_64-linux-gnu/zlib.h ]
}

make_ok() { have make || have ninja; }

# Put a distro MPI whose wrappers are not in $PATH (e.g. Fedora openmpi-devel)
# onto $PATH without a module system.
seed_mpi_path() {
    if mpi_ok; then
        return 0
    fi
    local d
    for d in \
        /usr/lib64/openmpi/bin /usr/lib/openmpi/bin \
        /usr/lib/x86_64-linux-gnu/openmpi/bin \
        /usr/lib64/mpich/bin /usr/lib/mpich/bin \
        /usr/lib/x86_64-linux-gnu/mpich/bin; do
        if [ -x "$d/mpicc" ]; then
            PATH="$d:$PATH"
            export PATH
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# library detection (header/config version, not just directory existence)
# ---------------------------------------------------------------------------
config_version() {
    # Read PACKAGE_VERSION from the first CMake *ConfigVersion.cmake under ROOT.
    local root="$1" f
    for f in "$root"/lib/cmake/*/*ConfigVersion.cmake \
        "$root"/lib*/cmake/*/*ConfigVersion.cmake \
        "$root"/lib*/cmake/*/*-config-version.cmake \
        "$root"/share/cmake/*/*ConfigVersion.cmake; do
        [ -f "$f" ] || continue
        sed -n 's/.*PACKAGE_VERSION "\([^"]*\)".*/\1/p' "$f" | head -n1
        return 0
    done
    return 1
}

header_version_three() {
    # header_version_three FILE PREFIX SUFFIX_MAJOR SUFFIX_MINOR SUFFIX_PATCH
    local file="$1" prefix="$2" major="$3" minor="$4" patch="$5"
    [ -f "$file" ] || return 1
    local a b c
    a="$(sed -n "s/^#define[[:space:]]*${prefix}${major}[[:space:]]*\([0-9]*\).*/\1/p" "$file" | head -n1)"
    b="$(sed -n "s/^#define[[:space:]]*${prefix}${minor}[[:space:]]*\([0-9]*\).*/\1/p" "$file" | head -n1)"
    c="$(sed -n "s/^#define[[:space:]]*${prefix}${patch}[[:space:]]*\([0-9]*\).*/\1/p" "$file" | head -n1)"
    [ -n "$a" ] && [ -n "$b" ] && [ -n "$c" ] || return 1
    printf '%s.%s.%s\n' "$a" "$b" "$c"
}

boost_version() {
    local f="$BOOST_ROOT/include/boost/version.hpp"
    [ -f "$f" ] || return 1
    sed -n 's/.*#define BOOST_LIB_VERSION "\([0-9]*\)_\([0-9]*\)".*/\1.\2/p' "$f" | head -n1
}
boost_ok() { version_ge "$(boost_version || true)" "1.74"; }

blosc_version() {
    local f="$BLOSC_ROOT/include/blosc2.h"
    header_version_three "$f" "BLOSC2_VERSION_" MAJOR MINOR RELEASE
}
blosc_ok() { version_ge "$(blosc_version || true)" "2.0"; }

libpng_version() {
    local f="$LIBPNG_ROOT/include/png.h"
    [ -f "$f" ] || return 1
    sed -n 's/.*#define PNG_LIBPNG_VER_STRING "\([^"]*\)".*/\1/p' "$f" | head -n1
}
libpng_ok() { version_ge "$(libpng_version || true)" "1.2.9"; }

pngwriter_version() {
    local v
    v="$(config_version "$PNGwriter_ROOT" || true)"
    [ -n "$v" ] && {
        printf '%s\n' "$v"
        return 0
    }
    # PNGwriter 0.7.0 does not version its public header; the config file is
    # authoritative, so presence of the header is the fallback.
    [ -f "$PNGwriter_ROOT/include/pngwriter.h" ] && printf '0.7.0\n'
}
pngwriter_ok() { version_ge "$(pngwriter_version || true)" "0.7.0"; }

hdf5_version() {
    header_version_three "$HDF5_ROOT/include/hdf5.h" "H5_VERS_" MAJOR MINOR RELEASE
}
hdf5_ok() { version_ge "$(hdf5_version || true)" "1.10"; }

adios2_version() {
    local v
    v="$(config_version "$ADIOS2_ROOT" || true)"
    [ -n "$v" ] && {
        printf '%s\n' "$v"
        return 0
    }
    header_version_three "$ADIOS2_ROOT/include/adios2/common/ADIOSConfig.h" "ADIOS2_VERSION_" MAJOR MINOR PATCH
}
adios2_ok() { version_ge "$(adios2_version || true)" "2.10"; }

openpmd_version() {
    local f="$OPENPMD_ROOT/include/openPMD/version.hpp"
    header_version_three "$f" "OPENPMDAPI_VERSION_" MAJOR MINOR PATCH
}
openpmd_ok() { version_ge "$(openpmd_version || true)" "0.17"; }

fftw_version() {
    local f
    for f in "$FFTW_ROOT"/lib*/pkgconfig/fftw3.pc "$FFTW_ROOT"/lib/pkgconfig/fftw3.pc; do
        [ -f "$f" ] || continue
        sed -n 's/^Version:[[:space:]]*//p' "$f" | head -n1
        return 0
    done
    return 1
}
fftw_ok() { version_ge "$(fftw_version || true)" "3.3"; }

# ---------------------------------------------------------------------------
# package manager (toolchain/basics only)
# ---------------------------------------------------------------------------
PM="none"
detect_pm() {
    local candidate
    for candidate in apt-get dnf pacman zypper; do
        if have "$candidate"; then
            PM="$candidate"
            return 0
        fi
    done
    PM="none"
}

pm_install() {
    [ "$#" -gt 0 ] || return 0
    detect_pm
    log "installing system packages: $*"
    if [ "$MODE" = "dry-run" ]; then
        return 0
    fi
    local sudo_cmd=""
    if [ "$(id -u)" -ne 0 ]; then
        if have sudo; then
            sudo_cmd="sudo"
        else
            die "cannot install system packages ($*) without root/sudo; install them manually"
        fi
    fi
    case "$PM" in
        apt-get)
            $sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get update -qq
            $sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
            ;;
        dnf)
            $sudo_cmd dnf install -y "$@"
            ;;
        pacman)
            $sudo_cmd pacman -S --noconfirm --needed "$@"
            ;;
        zypper)
            $sudo_cmd zypper --non-interactive install "$@"
            ;;
        *)
            die "no supported package manager found; install these manually: $*"
            ;;
    esac
}

pkg_compiler() {
    case "$PM" in
        apt-get) printf 'build-essential\n' ;;
        dnf) printf 'gcc gcc-c++ make\n' ;;
        pacman) printf 'base-devel\n' ;;
        zypper) printf 'gcc gcc-c++ make\n' ;;
    esac
}
pkg_utils() {
    case "$PM" in
        apt-get) printf 'git curl ca-certificates wget tar bzip2 pkg-config\n' ;;
        dnf) printf 'git curl ca-certificates wget tar bzip2 pkgconf-pkg-config\n' ;;
        pacman) printf 'git curl wget tar bzip2 pkgconf\n' ;;
        zypper) printf 'git curl ca-certificates wget tar bzip2 pkg-config\n' ;;
    esac
}
pkg_zlib() {
    case "$PM" in
        apt-get) printf 'zlib1g-dev\n' ;;
        dnf) printf 'zlib-devel\n' ;;
        pacman) printf 'zlib\n' ;;
        zypper) printf 'zlib-devel\n' ;;
    esac
}
pkg_mpi() {
    case "$PM" in
        apt-get) printf 'libopenmpi-dev openmpi-bin\n' ;;
        dnf) printf 'openmpi-devel\n' ;;
        pacman) printf 'openmpi\n' ;;
        zypper) printf 'openmpi-devel\n' ;;
    esac
}
pkg_cmake() {
    case "$PM" in
        apt-get | dnf | pacman | zypper) printf 'cmake\n' ;;
    esac
}

# ---------------------------------------------------------------------------
# download/build helpers
# ---------------------------------------------------------------------------
fetch() { # URL DEST
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        return 0
    fi
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

untar() { # ARCHIVE -> prints source dir
    local archive="$1" dest="$2"
    mkdir -p "$dest"
    tar -xzf "$archive" -C "$dest"
}

export_cmake_prefix() {
    CMAKE_PREFIX_PATH="$1:${CMAKE_PREFIX_PATH:-}"
    LD_LIBRARY_PATH="$1/lib:$1/lib64:${LD_LIBRARY_PATH:-}"
    CPATH="$1/include:${CPATH:-}"
    export CMAKE_PREFIX_PATH LD_LIBRARY_PATH CPATH
}

build_cmake() {
    local archive="$SOURCE_DIR/cmake-$CMAKE_VERSION.tar.gz"
    fetch "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    (
        cd "$SOURCE_DIR/cmake-$CMAKE_VERSION"
        ./bootstrap --prefix="$CMAKE_ROOT" --parallel="$JOBS"
        make -j "$JOBS"
        make install
    )
    PATH="$CMAKE_ROOT/bin:$PATH"
    export PATH
}

install_boost() {
    local tag="${BOOST_VERSION//./_}"
    local archive="$SOURCE_DIR/boost_$tag.tar.gz"
    fetch "https://archives.boost.io/release/$BOOST_VERSION/source/boost_$tag.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    (
        cd "$SOURCE_DIR/boost_$tag"
        ./bootstrap.sh \
            --with-libraries=atomic,chrono,context,date_time,fiber,filesystem,math,program_options,serialization,system,thread \
            --prefix="$BOOST_ROOT" \
            CC="$(command -v cc)" CXX="$(command -v c++)"
        ./b2 cxxflags="-std=c++20" -j "$JOBS"
        ./b2 install
    )
    export_cmake_prefix "$BOOST_ROOT"
}

install_blosc() {
    local archive="$SOURCE_DIR/c-blosc2-${BLOSC_VERSION}.tar.gz"
    fetch "https://github.com/Blosc/c-blosc2/archive/refs/tags/v${BLOSC_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    cmake -S "$SOURCE_DIR/c-blosc2-${BLOSC_VERSION}" -B "$SOURCE_DIR/c-blosc2-${BLOSC_VERSION}-build" \
        -DCMAKE_INSTALL_PREFIX="$BLOSC_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPREFER_EXTERNAL_ZLIB=ON \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    cmake --build "$SOURCE_DIR/c-blosc2-${BLOSC_VERSION}-build" -j "$JOBS" --target install
    export_cmake_prefix "$BLOSC_ROOT"
}

install_libpng() {
    local archive="$SOURCE_DIR/libpng-${LIBPNG_VERSION}.tar.gz"
    fetch "https://ftp-osl.osuosl.org/pub/libpng/src/libpng16/libpng-${LIBPNG_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    (
        cd "$SOURCE_DIR/libpng-${LIBPNG_VERSION}"
        ./configure --prefix="$LIBPNG_ROOT" --enable-static --enable-shared
        make -j "$JOBS"
        make install
    )
    export_cmake_prefix "$LIBPNG_ROOT"
}

install_pngwriter() {
    local archive="$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}.tar.gz"
    fetch "https://github.com/pngwriter/pngwriter/archive/refs/tags/${PNGWRITER_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    cmake -S "$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}" -B "$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}-build" \
        -DCMAKE_INSTALL_PREFIX="$PNGwriter_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DPNGwriter_USE_FREETYPE=OFF \
        -DPNG_ROOT="$LIBPNG_ROOT" \
        -DCMAKE_PREFIX_PATH="$LIBPNG_ROOT;${CMAKE_PREFIX_PATH:-}"
    cmake --build "$SOURCE_DIR/pngwriter-${PNGWRITER_VERSION}-build" -j "$JOBS" --target install
    export_cmake_prefix "$PNGwriter_ROOT"
}

install_hdf5() {
    local archive="$SOURCE_DIR/hdf5-${HDF5_VERSION}.tar.gz"
    fetch "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_${HDF5_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    cmake -S "$SOURCE_DIR/hdf5-hdf5_${HDF5_VERSION}" -B "$SOURCE_DIR/hdf5-${HDF5_VERSION}-build" \
        -DCMAKE_INSTALL_PREFIX="$HDF5_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDF5_ENABLE_PARALLEL=ON \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_C_COMPILER="${MPI_CC:-mpicc}" \
        -DCMAKE_CXX_COMPILER="${MPI_CXX:-mpicxx}"
    cmake --build "$SOURCE_DIR/hdf5-${HDF5_VERSION}-build" -j "$JOBS" --target install
    export_cmake_prefix "$HDF5_ROOT"
}

install_adios2() {
    local archive="$SOURCE_DIR/adios2-${ADIOS2_VERSION}.tar.gz"
    fetch "https://github.com/ornladios/ADIOS2/archive/refs/tags/v${ADIOS2_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    cmake -S "$SOURCE_DIR/ADIOS2-${ADIOS2_VERSION}" -B "$SOURCE_DIR/adios2-${ADIOS2_VERSION}-build" \
        -DCMAKE_INSTALL_PREFIX="$ADIOS2_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DADIOS2_BUILD_EXAMPLES=OFF \
        -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_BZip2=OFF \
        -DADIOS2_USE_MPI=ON \
        -DADIOS2_USE_HDF5=ON \
        -DHDF5_ROOT="$HDF5_ROOT" \
        -DCMAKE_C_COMPILER="${MPI_CC:-mpicc}" \
        -DCMAKE_CXX_COMPILER="${MPI_CXX:-mpicxx}" \
        -DCMAKE_PREFIX_PATH="$HDF5_ROOT;${CMAKE_PREFIX_PATH:-}"
    cmake --build "$SOURCE_DIR/adios2-${ADIOS2_VERSION}-build" -j "$JOBS" --target install
    export_cmake_prefix "$ADIOS2_ROOT"
}

install_openpmd() {
    local archive="$SOURCE_DIR/openPMD-api-${OPENPMD_VERSION}.tar.gz"
    fetch "https://github.com/openPMD/openPMD-api/archive/refs/tags/${OPENPMD_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    cmake -S "$SOURCE_DIR/openPMD-api-${OPENPMD_VERSION}" -B "$SOURCE_DIR/openpmd-${OPENPMD_VERSION}-build" \
        -DCMAKE_INSTALL_PREFIX="$OPENPMD_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DopenPMD_USE_HDF5=ON \
        -DopenPMD_USE_ADIOS2=ON \
        -DopenPMD_USE_MPI=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_ROOT="$HDF5_ROOT" \
        -DADIOS2_ROOT="$ADIOS2_ROOT" \
        -DCMAKE_C_COMPILER="${MPI_CC:-mpicc}" \
        -DCMAKE_CXX_COMPILER="${MPI_CXX:-mpicxx}" \
        -DCMAKE_PREFIX_PATH="$HDF5_ROOT;$ADIOS2_ROOT;${CMAKE_PREFIX_PATH:-}"
    cmake --build "$SOURCE_DIR/openpmd-${OPENPMD_VERSION}-build" -j "$JOBS" --target install
    export_cmake_prefix "$OPENPMD_ROOT"
}

install_fftw() {
    local archive="$SOURCE_DIR/fftw-${FFTW_VERSION}.tar.gz"
    fetch "http://fftw.org/fftw-${FFTW_VERSION}.tar.gz" "$archive"
    untar "$archive" "$SOURCE_DIR"
    (
        cd "$SOURCE_DIR/fftw-${FFTW_VERSION}"
        ./configure --prefix="$FFTW_ROOT" --enable-shared --enable-static
        make -j "$JOBS"
        make install
    )
    export_cmake_prefix "$FFTW_ROOT"
}

# ---------------------------------------------------------------------------
# status / plan / install
# ---------------------------------------------------------------------------
status_line() { # LABEL ok|missing DETAIL
    if [ "$2" = "ok" ]; then
        printf '  [ok]      %-14s %s\n' "$1" "$3"
    else
        printf '  [MISSING] %-14s %s\n' "$1" "$3"
    fi
}

lib_status() { # LABEL CHECK_FUNC ROOT VERSION_FUNC MIN
    local label="$1" check="$2" root="$3" vfunc="$4" min="$5" v=""
    if "$check"; then
        v="$($vfunc || true)"
        status_line "$label" ok "${v:-present} ($root)"
        return 0
    fi
    if [ -d "$root" ]; then
        v="$($vfunc || true)"
        status_line "$label" missing "found ${v:-unversioned} at $root, need >=$min"
    else
        status_line "$label" missing "not installed (wanted >=$min at $root)"
    fi
    return 1
}

do_check() {
    local rc=0
    printf 'Dependency status (PIC_LIBS=%s)\n' "$PIC_LIBS"
    if compiler_ok; then
        status_line "compiler" ok "$(compiler_version)"
    else
        status_line "compiler" missing "need GCC >=10 or Clang >=11"
        rc=1
    fi
    if cmake_ok; then
        status_line "cmake" ok "$(cmake_version)"
    else
        status_line "cmake" missing "need >=$CMAKE_MIN_VERSION, found $(cmake_version || echo none)"
        rc=1
    fi
    seed_mpi_path || true
    if mpi_ok; then
        status_line "mpi" ok "$(mpi_version || echo present)"
    else
        status_line "mpi" missing "need mpicc/mpicxx/mpiexec"
        rc=1
    fi
    if zlib_ok; then
        status_line "zlib" ok "$(pkg-config --modversion zlib 2>/dev/null || echo present)"
    else
        status_line "zlib" missing "need zlib headers/libs"
        rc=1
    fi
    lib_status "boost" boost_ok "$BOOST_ROOT" boost_version "1.74" || rc=1
    lib_status "c-blosc2" blosc_ok "$BLOSC_ROOT" blosc_version "2.0" || rc=1
    lib_status "libpng" libpng_ok "$LIBPNG_ROOT" libpng_version "1.2.9" || rc=1
    lib_status "PNGwriter" pngwriter_ok "$PNGwriter_ROOT" pngwriter_version "0.7.0" || rc=1
    lib_status "HDF5" hdf5_ok "$HDF5_ROOT" hdf5_version "1.10" || rc=1
    lib_status "ADIOS2" adios2_ok "$ADIOS2_ROOT" adios2_version "2.10" || rc=1
    lib_status "openPMD-api" openpmd_ok "$OPENPMD_ROOT" openpmd_version "0.17" || rc=1
    lib_status "FFTW" fftw_ok "$FFTW_ROOT" fftw_version "3.3" || rc=1
    if [ "$rc" -eq 0 ]; then
        printf '\nAll dependencies are satisfied.\n'
    else
        printf '\nSome dependencies are missing; run without --check (or `pic-deps install`) to build them.\n'
    fi
    return "$rc"
}

plan_line() { # LABEL NEEDS_ACTION DETAIL
    if [ "$2" = "no" ]; then
        printf '  [skip]    %-14s %s\n' "$1" "$3"
    else
        printf '  [install] %-14s %s\n' "$1" "$3"
    fi
}

do_dry_run() {
    printf 'Install plan (PIC_LIBS=%s, build dir %s)\n' "$PIC_LIBS" "$SOURCE_DIR"
    if compiler_ok; then plan_line compilers no "$(compiler_version)"; else plan_line compilers yes "distro toolchain (may need sudo)"; fi
    if cmake_ok; then
        plan_line cmake no "$(cmake_version)"
    elif have cmake; then
        plan_line cmake yes "distro CMake $(cmake_version) is < $CMAKE_MIN_VERSION; build $CMAKE_VERSION"
    else
        plan_line cmake yes "install distro CMake, else build $CMAKE_VERSION from source"
    fi
    seed_mpi_path || true
    if mpi_ok; then plan_line mpi no "$(mpi_version || echo present)"; else plan_line mpi yes "distro OpenMPI/MPICH"; fi
    if zlib_ok; then plan_line zlib no present; else plan_line zlib yes "distro zlib"; fi
    local dep label check root vfunc min
    for dep in boost:Boost blosc:c-blosc2 libpng:libpng pngwriter:PNGwriter hdf5:HDF5 adios2:ADIOS2 openpmd:openPMD-api fftw:FFTW; do
        label="${dep#*:}"
        case "$label" in
            Boost) check=boost_ok root="$BOOST_ROOT" vfunc=boost_version min="$BOOST_VERSION" ;;
            c-blosc2) check=blosc_ok root="$BLOSC_ROOT" vfunc=blosc_version min="$BLOSC_VERSION" ;;
            libpng) check=libpng_ok root="$LIBPNG_ROOT" vfunc=libpng_version min="$LIBPNG_VERSION" ;;
            PNGwriter) check=pngwriter_ok root="$PNGwriter_ROOT" vfunc=pngwriter_version min="$PNGWRITER_VERSION" ;;
            HDF5) check=hdf5_ok root="$HDF5_ROOT" vfunc=hdf5_version min="$HDF5_VERSION" ;;
            ADIOS2) check=adios2_ok root="$ADIOS2_ROOT" vfunc=adios2_version min="$ADIOS2_VERSION" ;;
            openPMD-api) check=openpmd_ok root="$OPENPMD_ROOT" vfunc=openpmd_version min="$OPENPMD_VERSION" ;;
            FFTW) check=fftw_ok root="$FFTW_ROOT" vfunc=fftw_version min="$FFTW_VERSION" ;;
        esac
        if is_enabled "${dep%%:*}"; then
            if "$check"; then
                plan_line "$label" no "$($vfunc || true) at $root"
            else
                plan_line "$label" yes "build $min into $root"
            fi
        fi
    done
    printf '\nNo changes made (--dry-run).\n'
}

ensure_toolchain() {
    if ! compiler_ok; then
        log "C++ compiler missing/too old -> installing distro toolchain"
        pm_install $(pkg_compiler)
        compiler_ok || die "still no C++ compiler >= GCC 10 / Clang 11 after installation"
    fi
    if ! have git || ! have tar || ! have bzip2 || { ! have curl && ! have wget; }; then
        log "build utilities missing -> installing"
        pm_install $(pkg_utils)
    fi
    if ! make_ok; then
        log "make/ninja missing -> installing"
        pm_install $(pkg_compiler)
    fi
    if ! zlib_ok; then
        log "zlib missing -> installing"
        pm_install $(pkg_zlib)
        zlib_ok || die "zlib is still missing after installation"
    fi
    seed_mpi_path || true
    if ! mpi_ok; then
        log "MPI wrappers missing -> installing distro MPI"
        pm_install $(pkg_mpi)
        seed_mpi_path || true
        mpi_ok || die "MPI wrappers are still missing after installation"
    fi
    if ! cmake_ok; then
        if have cmake; then
            log "distro CMake $(cmake_version) is older than $CMAKE_MIN_VERSION -> trying a newer package"
        else
            log "CMake missing -> installing distro CMake"
        fi
        pm_install $(pkg_cmake)
    fi
    if ! cmake_ok; then
        log "distro CMake still older than $CMAKE_MIN_VERSION -> building CMake $CMAKE_VERSION into $CMAKE_ROOT"
        build_cmake
        cmake_ok || die "CMake build did not provide >= $CMAKE_MIN_VERSION"
    fi
    export MPI_CC="${MPI_CC:-$(command -v mpicc)}"
    export MPI_CXX="${MPI_CXX:-$(command -v mpicxx)}"
}

check_selected() {
    # Like do_check, but only for the dependencies selected with --only.
    local rc=0 dep key label check root vfunc min
    for dep in boost:Boost blosc:c-blosc2 libpng:libpng pngwriter:PNGwriter hdf5:HDF5 adios2:ADIOS2 openpmd:openPMD-api fftw:FFTW; do
        key="${dep%%:*}"
        label="${dep#*:}"
        case "$key" in
            boost) check=boost_ok root="$BOOST_ROOT" vfunc=boost_version min="$BOOST_VERSION" ;;
            blosc) check=blosc_ok root="$BLOSC_ROOT" vfunc=blosc_version min="$BLOSC_VERSION" ;;
            libpng) check=libpng_ok root="$LIBPNG_ROOT" vfunc=libpng_version min="$LIBPNG_VERSION" ;;
            pngwriter) check=pngwriter_ok root="$PNGwriter_ROOT" vfunc=pngwriter_version min="$PNGWRITER_VERSION" ;;
            hdf5) check=hdf5_ok root="$HDF5_ROOT" vfunc=hdf5_version min="$HDF5_VERSION" ;;
            adios2) check=adios2_ok root="$ADIOS2_ROOT" vfunc=adios2_version min="$ADIOS2_VERSION" ;;
            openpmd) check=openpmd_ok root="$OPENPMD_ROOT" vfunc=openpmd_version min="$OPENPMD_VERSION" ;;
            fftw) check=fftw_ok root="$FFTW_ROOT" vfunc=fftw_version min="$FFTW_VERSION" ;;
        esac
        is_enabled "$key" || continue
        lib_status "$label" "$check" "$root" "$vfunc" "$min" || rc=1
    done
    return "$rc"
}

do_install() {
    mkdir -p "$PIC_LIBS" "$SOURCE_DIR"
    detect_pm
    ensure_toolchain

    # Keep the build tree unborn until now: the profile's roots are what
    # pic-deps check enumerates, so always build into $PIC_LIBS.
    if ! is_enabled boost; then
        :
    elif [ ! -d "$BOOST_ROOT" ] || ! boost_ok; then
        log "Installing Boost $BOOST_VERSION into $BOOST_ROOT"
        install_boost
    else
        log "Boost $(boost_version) already satisfied at $BOOST_ROOT (skipping)"
    fi

    if ! is_enabled blosc; then
        :
    elif [ ! -d "$BLOSC_ROOT" ] || ! blosc_ok; then
        log "Installing c-blosc2 $BLOSC_VERSION into $BLOSC_ROOT"
        install_blosc
    else
        log "c-blosc2 $(blosc_version) already satisfied at $BLOSC_ROOT (skipping)"
    fi

    if ! is_enabled libpng; then
        :
    elif [ ! -d "$LIBPNG_ROOT" ] || ! libpng_ok; then
        log "Installing libpng $LIBPNG_VERSION into $LIBPNG_ROOT"
        install_libpng
    else
        log "libpng $(libpng_version) already satisfied at $LIBPNG_ROOT (skipping)"
    fi

    if ! is_enabled pngwriter; then
        :
    elif [ ! -d "$PNGwriter_ROOT" ] || ! pngwriter_ok; then
        log "Installing PNGwriter $PNGWRITER_VERSION into $PNGwriter_ROOT"
        install_pngwriter
    else
        log "PNGwriter $(pngwriter_version) already satisfied at $PNGwriter_ROOT (skipping)"
    fi

    if ! is_enabled hdf5; then
        :
    elif [ ! -d "$HDF5_ROOT" ] || ! hdf5_ok; then
        log "Installing parallel HDF5 $HDF5_VERSION into $HDF5_ROOT"
        install_hdf5
    else
        log "HDF5 $(hdf5_version) already satisfied at $HDF5_ROOT (skipping)"
    fi

    if ! is_enabled adios2; then
        :
    elif [ ! -d "$ADIOS2_ROOT" ] || ! adios2_ok; then
        log "Installing ADIOS2 $ADIOS2_VERSION into $ADIOS2_ROOT"
        install_adios2
    else
        log "ADIOS2 $(adios2_version) already satisfied at $ADIOS2_ROOT (skipping)"
    fi

    if ! is_enabled openpmd; then
        :
    elif [ ! -d "$OPENPMD_ROOT" ] || ! openpmd_ok; then
        log "Installing openPMD-api $OPENPMD_VERSION into $OPENPMD_ROOT"
        install_openpmd
    else
        log "openPMD-api $(openpmd_version) already satisfied at $OPENPMD_ROOT (skipping)"
    fi

    if ! is_enabled fftw; then
        :
    elif [ ! -d "$FFTW_ROOT" ] || ! fftw_ok; then
        log "Installing FFTW $FFTW_VERSION into $FFTW_ROOT"
        install_fftw
    else
        log "FFTW $(fftw_version) already satisfied at $FFTW_ROOT (skipping)"
    fi

    # Re-check and fail if a genuine installation problem was hidden by the
    # per-dependency early exits above. With --only, only the selected
    # dependencies are required to be present.
    local check_ok=1
    if [ -n "$ONLY" ]; then
        printf 'Selected dependency status after install:\n'
        check_selected || check_ok=0
    else
        do_check || check_ok=0
    fi
    if [ "$check_ok" -eq 1 ]; then
        printf '\nDependencies are ready. Re-run `pic-deps check` to verify from the profile.\n'
        printf 'Delete the build tree when you no longer need it:\n'
        printf '    rm -rf %s\n' "$SOURCE_DIR"
    else
        die "some dependencies are still missing after the install run"
    fi
}

case "$MODE" in
    check)
        do_check
        ;;
    dry-run)
        do_dry_run
        ;;
    install)
        do_install
        ;;
esac

# ---------------------------------------------------------------------------
# Alternative installers (documented variants; not executed by this script)
#
# The same dependency set can be obtained with Spack or Conda on machines where
# those are preferred. Both are reported to work well upstream. The important
# constraint is identical to this script: point the profile's PIC_LIBS/*_ROOT
# exports at the resulting prefix, or install into PIC_LIBS with `--prefix`.
#
# Spack (recommended by docs/source/install/dependencies.rst for HPC):
#
#   . /path/to/spack/share/spack/setup-env.sh
#   spack install gcc@12 %system-compiler          # optional newer compiler
#   spack install cmake@3.31 %gcc@12
#   spack install openmpi %gcc@12
#   spack install boost@1.87.0 +program_options %gcc@12 ^openmpi
#   spack install hdf5@2.0.0 +mpi +cxx %gcc@12 ^openmpi
#   spack install adios2@2.11.0 +mpi +hdf5 +blosc %gcc@12 ^hdf5+mpi ^openmpi
#   spack install openpmd-api@0.17.1 +mpi +hdf5 +adios2 %gcc@12 ^adios2+hdf5
#   spack install fftw@3.3.10 %gcc@12
#   spack install c-blosc2@2.22.0 %gcc@12
#   spack install libpng@1.6.34 %gcc@12
#   spack install pngwriter@0.7.0 %gcc@12 ^libpng
#   spack load --sh cmake openmpi boost hdf5 adios2 openpmd-api fftw c-blosc2 libpng pngwriter
#
# To keep the guard-root contract without rewriting the profile, install with
# Spack's prefix layout and export the roots, e.g.:
#   export PIC_LIBS=/opt/pic-libs
#   spack install --prefix="$PIC_LIBS/boost/$BOOST_VERSION" boost@1.87.0 +program_options
#   ... (one spack install --prefix per dependency, matching the *_ROOT paths)
#
# Conda/micromamba (conda-forge provides all but PNGwriter; build PNGwriter
# against the conda libpng with the normal CMake invocation):
#
#   micromamba create -p "$PIC_LIBS/conda" -c conda-forge \
#       'gxx_linux-64>=10' 'cmake>=3.28' openmpi \
#       'boost>=1.87' hdf5='*=mpi*' adios2='*=mpi*' openpmd-api \
#       fftw c-blosc2 libpng zlib pkg-config make git curl
#   micromamba activate "$PIC_LIBS/conda"
#   export CONDA_PREFIX="$PIC_LIBS/conda"
#   export BOOST_ROOT="$CONDA_PREFIX" BLOSC_ROOT="$CONDA_PREFIX" \
#          LIBPNG_ROOT="$CONDA_PREFIX" HDF5_ROOT="$CONDA_PREFIX" \
#          ADIOS2_ROOT="$CONDA_PREFIX" OPENPMD_ROOT="$CONDA_PREFIX" \
#          FFTW_ROOT="$CONDA_PREFIX"
#   # PNGwriter is not on conda-forge:
#   cmake -S pngwriter -B pngwriter-build \
#         -DCMAKE_INSTALL_PREFIX="$PIC_LIBS/pngwriter/$PNGWRITER_VERSION" \
#         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
#         -DPNGwriter_USE_FREETYPE=OFF && \
#     cmake --build pngwriter-build -j "$(nproc)" --target install
#   export PNGwriter_ROOT="$PIC_LIBS/pngwriter/$PNGWRITER_VERSION"
#
# On both variants export CMAKE_PREFIX_PATH/LD_LIBRARY_PATH/CPATH the same way
# the bash profile does, then run `pic-deps check` to confirm the root contract.
# ---------------------------------------------------------------------------
