#!/usr/bin/env zsh
export PIC_PROFILE=$(cd $(dirname ${(%):-%N}) && pwd)"/"$(basename ${(%):-%N})

export PIC_BACKEND="omp2b:native" # running on cpu
export PIC_SYSTEM_TEMPLATE_PATH=${PIC_SYSTEM_TEMPLATE_PATH:-"etc/picongpu/zsh"}

export PICSRC="{{{pic_src_path}}}"

export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
export PATH=$PICSRC/bin:$PATH
export PATH=$PICSRC/src/tools/bin:$PATH

# "tbg" default options #######################################################
export TBG_SUBMIT="zsh"
export TBG_TPLFILE="/home/lenz/profiles/mpirun.tpl"

# Handling spack is kind of non-trivial here:
source /home/lenz/opt/spack/share/spack/setup-env.sh
# Somehow this view breaks my terminal, so instead of using it directly,
# I'll just look up what it loads and load that manually.
spack env activate picongpu --with-view=default
packages="$(spack find --loaded 2>&1 | sed -n 's/\[+\] //gp' | sort | uniq)"
spack load $packages
