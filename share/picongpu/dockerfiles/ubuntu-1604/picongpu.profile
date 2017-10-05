# Text Editor for Tools #######################################################
#   - examples: "nano", "vim", "emacs -nw", "vi" or without terminal: "gedit"
#export EDITOR="nano"

# Modules #####################################################################
#
. /usr/share/lmod/5.8/init/bash
. $HOME/src/spack/share/spack/setup-env.sh
export MODULEPATH=$HOME/src/spack/share/spack/lmod/linux-ubuntu16-x86_64/Core

# Core Dependencies (based on gcc 5.4.0)
spack load cmake
spack load boost
spack load cuda
spack load openmpi

# Plugins (optional)
spack load zlib libpng freetype pngwriter
spack load hdf5 libsplash
spack load libjpeg-turbo jansson icet isaac
spack load isaac-server

# either use libSplash or ADIOS for file I/O
#spack load adios

# Debug Tools
#spack load gdb
#spack load valgrind

# Environment #################################################################
#
export PICSRC=/home/$(whoami)/src/picongpu
export PIC_BACKEND="cuda"
export PIC_PROFILE=$(cd $(dirname $BASH_SOURCE) && pwd)"/"$(basename $BASH_SOURCE)

# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
export MY_MAILNOTIFY="n"
export MY_MAIL="someone@example.com"
export MY_NAME="$(whoami) <$MY_MAIL>"

export PATH=$PATH:$PICSRC
export PATH=$PATH:$PICSRC/src/splash2txt/build
export PATH=$PATH:$PICSRC/src/tools/bin

export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH

# "tbg" default options #######################################################
#   - interactive (bash + mpiexec)
export TBG_SUBMIT="bash"
export TBG_TPLFILE="submit/bash/bash_mpiexec.tpl"
