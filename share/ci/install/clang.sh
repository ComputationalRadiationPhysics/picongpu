#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

clang_version=$(echo $CXX_VERSION | tr -d "clang++-")
echo "Clang-version: $clang_version"

# the containers 4.1 do not ship an llvm apt repository, only the clang
# version provided by the distribution is available (ubuntu 24.04: up to
# clang 20, ubuntu 22.04: up to clang 15), all other versions are provided by
# apt.llvm.org
apt -y update
# check for an installable candidate, 'apt-cache show' also succeeds with empty
# output for package names which are known to apt but have no candidate, e.g.
# clang-21 in ubuntu 24.04
clang_candidate=$(apt-cache policy clang-${clang_version} 2>/dev/null | awk '/Candidate:/ { print $2 }')
if [ -z "${clang_candidate}" ] || [ "${clang_candidate}" = "(none)" ] ; then
    ubuntu_code_name=$(. /etc/os-release && echo "${VERSION_CODENAME}")
    echo "clang-${clang_version} is not provided by the container, adding the apt.llvm.org repository"
    apt install -y ca-certificates curl gnupg
    install -d -m 0755 /etc/apt/keyrings
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /tmp/apt.llvm.org.asc
    gpg --dearmor --yes -o /etc/apt/keyrings/apt.llvm.org.gpg /tmp/apt.llvm.org.asc
    chmod 0644 /etc/apt/keyrings/apt.llvm.org.gpg
    echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/${ubuntu_code_name}/ llvm-toolchain-${ubuntu_code_name}-${clang_version} main" > /etc/apt/sources.list.d/apt.llvm.org.list
    apt -y update
fi
apt install -y clang-${clang_version}
if [[ "$PIC_BACKEND" =~ omp2b.* ]] ; then
    # the containers ship a fixed libomp version (currently 18) whose
    # runtime conflicts with any other versioned libomp, so the shipped
    # one has to be removed before the matching package can be installed
    installed_libomp=$(dpkg-query -W -f='${Package} ${Status}\n' 'libomp*' 2>/dev/null \
        | awk '$4 == "installed" { print $1 }' || true)
    if [ -n "$installed_libomp" ] ; then
        apt remove -y ${installed_libomp}
    fi
    apt install -y libomp-${clang_version}-dev
fi

# export the selected compiler so that all following build steps default to it
export CXX=${CXX_VERSION}
export CC=clang-${clang_version}
