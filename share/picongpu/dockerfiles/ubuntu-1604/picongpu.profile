# Modules #####################################################################
#
. /usr/share/lmod/5.8/init/bash
. $HOME/src/spack/share/spack/setup-env.sh
export MODULEPATH=$HOME/src/spack/share/spack/lmod/linux-ubuntu16-x86_64/Core

# Core Dependencies (based on gcc 5.4.0)
module load cmake
module load boost
module load cuda
module load openmpi

# Plugins (optional)
module load zlib libpng freetype pngwriter
module load hdf5 libsplash
module load libjpeg-turbo jansson icet isaac
module load isaac-server

# either use libSplash or ADIOS for file I/O
#module load adios

# Debug Tools
#module load gdb
#module load valgrind

# Environment #################################################################
#
export PICSRC=/home/$(whoami)/src/picongpu
export PIC_PROFILE=$(cd $(dirname $BASH_SOURCE) && pwd)"/"$(basename $BASH_SOURCE)

# send me mails on job (b)egin, (e)nd, (a)bortion or (n)o mail
export MY_MAILNOTIFY="n"
export MY_MAIL="someone@example.com"
export MY_NAME="$(whoami) <$MY_MAIL>"

export PATH=$PATH:$PICSRC
export PATH=$PATH:$PICSRC/src/splash2txt/build
export PATH=$PATH:$PICSRC/src/tools/bin

export PYTHONPATH=$PICSRC/src/tools/lib/python:$PYTHONPATH

# "tbg" default options #######################################################
#   - interactive (bash + mpiexec)
export TBG_SUBMIT="bash"
export TBG_TPLFILE="submit/bash/bash_mpiexec.tpl"
