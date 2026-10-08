#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

clang_version=$(echo $CXX_VERSION | tr -d "clang++-")
echo "Clang-version: $clang_version"

# the containers version 4.0 ship apt sources for clang up to version 19, for
# newer versions the matching llvm-toolchain source has to be added
if [[ ${clang_version} -gt 19 ]] ; then
    ubuntu_code_name=$(. /etc/os-release && echo "${VERSION_CODENAME}")
    if ! grep -q "llvm-toolchain-${ubuntu_code_name}-${clang_version} " /etc/apt/sources.list.d/llvm.list 2>/dev/null ; then
        echo "deb http://apt.llvm.org/${ubuntu_code_name}/ llvm-toolchain-${ubuntu_code_name}-${clang_version} main" >> /etc/apt/sources.list.d/llvm.list
        echo "deb-src http://apt.llvm.org/${ubuntu_code_name}/ llvm-toolchain-${ubuntu_code_name}-${clang_version} main" >> /etc/apt/sources.list.d/llvm.list
    fi
fi

apt -y update

if ! agc-manager -e clang@${clang_version}; then
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
else
    CLANG_BASE_PATH="$(agc-manager -b clang@${clang_version})"
    export PATH=$CLANG_BASE_PATH/bin:$PATH
fi
