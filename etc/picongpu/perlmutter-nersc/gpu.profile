# Name and Path of this Script ############################### (DO NOT change!)
export PIC_PROFILE=$(cd $(dirname $BASH_SOURCE) && pwd)"/"$(basename $BASH_SOURCE)

# User Information ################################# (edit the following lines)
#   - automatically add your name and contact to output file meta data
#   - send me a mail on batch system jobs: NONE, BEGIN, END, FAIL, REQUEUE, ALL,
#     TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80 and/or TIME_LIMIT_50
export MY_MAILNOTIFY="NONE"
export MY_MAIL="someone@example.com"
export MY_NAME="$(whoami) <$MY_MAIL>"

# Project Information ######################################## (edit this line)
#   - project account for computing time
export proj="mXXXX"

# Text Editor for Tools ###################################### (edit this line)
#   - examples: "nano", "vim", "emacs -nw", "vi" or without terminal: "gedit"
#export EDITOR="nano"

# General modules #############################################################
#
module purge
module load PrgEnv-gnu/8.3.3
module load cmake/3.22.0
module load zlib/1.2.11
module load cudatoolkit/11.5
module load craype-accel-nvidia80
module load cray-hdf5-parallel/1.12.1.1

# Additional libraries
#
export PARTITION_LIB=$CFS/$proj/picongpu_libraries

# HDF5
HDF5_VERSION=1.12.1
export HDF5_ROOT=$PARTITION_LIB/HDF5/$HDF5_VERSION
export PATH=$HDF5_ROOT/bin:$PATH
export CPATH=$HDF5_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH
# BOOST (to include TR1)
BOOST_VERSION=1.74.0
export BOOST_ROOT=$PARTITION_LIB/BOOST/$BOOST_VERSION
export CPATH=$BOOST_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$BOST_ROOT/lib/cmake:$CMAKE_PREFIX_PATH

# PNGwriter
PNGWRITER_VERSION=0.7.0
PNGwriter_ROOT=$PARTITION_LIB/PNGWRITER/$PNGWRITER_VERSION
export CMAKE_PREFIX_PATH=$PNGWRITER_ROOT:$CMAKE_PREFIX_PATH

# C-BLOSC
BLOSC_VERSION=1.21.1
BLOSC_ROOT=$PARTITION_LIB/BLOSC/$BLOSC_VERSION
export CMAKE_PREFIX_PATH=$BLOSC_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$BLOSC_ROOT/lib:$LD_LIBRARY_PATH

# ADIOS2 with HDF5 and BLOSC support
ADIOS2_VERSION=2.7.1.436
ADIOS2_ROOT=$PARTITION_LIB/ADIOS2/$ADIOS2_VERSION
export PATH=$ADIOS2_ROOT/bin:$PATH
export CMAKE_PREFIX_PATH=$ADIOS2_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$ADIOS2_ROOT/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ADIOS2_ROOT/lib/python3.9:$PYTHONPATH

# openPMD API with ADIOS2, HDF5 support
OPENPMD_VERSION=0.14.4
OPENPMD_ROOT=$PARTITION_LIB/OPENPMD/$OPENPMD_VERSION
export PATH=$OPENPMD_ROOT/bin:$PATH
export CMAKE_PREFIX_PATH=$OPENPMD_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$OPENPMD_ROOT/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$OPENPMD_ROOT/lib64/python3.9:$PYTHONPATH

# Environment #################################################################
#
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which g++)

export PICSRC=$HOME/picongpu
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
export PIC_BACKEND="cuda:80"

export PATH=$PATH:$PICSRC
export PATH=$PATH:$PICSRC/bin
export PATH=$PATH:$PICSRC/src/tools/bin

export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH

# "tbg" default options #######################################################
#   - SLURM (sbatch)
#   - "defq" queue
export TBG_SUBMIT="sbatch"
export TBG_TPLFILE="etc/picongpu/perlmutter-nersc/gpu.tpl"
# allocate an interactive node for one hour to execute a mpi parallel application
#   getNode 2  # allocates two interactive A100 GPUs (default: 1)
function getNode() {
    if [ -z "$1" ] ; then
        numNodes=1
    else
        numNodes=$1
    fi
    echo "Hint: please use 'srun --cpu_bind=cores <COMMAND>' for launching multiple processes in the interactive mode."
    salloc --time=1:00:00 --nodes=$numNodes --ntasks-per-node 4 --gpus 4 --cpus-per-task=32 -A $proj -C gpu -q shared
}

# allocate an interactive device for one hour to execute a mpi parallel application
#   getDevice 2  # allocates two interactive devices (default: 1)
function getDevice() {
    if [ -z "$1" ] ; then
        numGPUs=1
    else
        if [ "$1" -gt 8 ] ; then
            echo "The maximal number of devices per node is 8." 1>&2
            return 1
        else
            numGPUs=$1
        fi
    fi
    echo "Hint: please use 'srun --cpu_bind=cores <COMMAND>' for launching multiple processes in the interactive mode."
    salloc --time=1:00:00 --ntasks-per-node=$numGPUs --cpus-per-task=16 --gpus=$numGPUs --mem=$((230000 / 4) * $numGPUs) -A $proj -C gpu -q shared
}

# allocate an interactive shell for compilation (without gpus)
function getShell() {
    srun --time=1:00:00 --nodes=1 --ntasks 1 --cpus-per-task=32 -A $proj -C gpu -q shared --pty bash
}

# Load autocompletion for PIConGPU commands
BASH_COMP_FILE=$PICSRC/bin/picongpu-completion.bash
if [ -f $BASH_COMP_FILE ] ; then
    source $BASH_COMP_FILE
else
    echo "bash completion file '$BASH_COMP_FILE' not found." >&2
fi
