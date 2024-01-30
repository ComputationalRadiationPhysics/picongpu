#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2023 PIConGPU contributors
# Authors: Mika Soren Voss, Hannes Wolf, Klaus Steiniger
# License: GPLv3+
#

function absolute_path()
{
    builtin cd -- $1 && pwd
}

help()
{
  echo "Execute a single particle simulation in a homogenious magnetic field to validate the pusher."
  echo "Simulation results are validated by checking the dependenc of the phaseshift caused by the pusher."
  echo ""
  echo "Usage: ci.sh"
  echo ""
  echo "Options"
  echo "-h | --help                   - show help"
  echo ""
}


#####################
## option handling ##
#####################
# options may be followed by
# - one colon to indicate they have a required argument
OPTS=`getopt -o h -l help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# parser
while true ; do
    case "$1" in
        -h|--help)
            echo -e "$(help)"
            shift
            exit 0
            ;;
        --) shift; break;;
    esac
    shift
done


############################
## build and run picongpu ##
############################

declare -a pathArray # Array for different simulation paths
j=5 # variable for calculation of the number of steps in a simulation run at a set time
for i in {0..4}
do
    j=$(($j-1))
        power=$((2**$j))
        revolutions=$((1000*$power))

if [ -d "./include" ] ; then
  pic-build -t $i
  ret_build=$?

else
  echo "Execute ci.sh from the directory where the simulation include dir is located!"
  exit 1
fi

if [ $ret_build -eq 0 ] ; then
  ## create simulation data directory
  simPath="./$i-simOutput"

  if [ -d "$simPath" ] ; then
      echo "Destination path already in use, cannot create new folder" >&2
      exit 1
  fi

  mkdir -p $simPath

  # use absolut path
  simPath=$(absolute_path $simPath)
  echo "$simPath"

  cd $simPath
  echo "Run!"
  mpiexec -n 1 ../bin/picongpu -d 1 1 1 -g 64 64 32 -s $revolutions --fieldBackground.duplicateFields --periodic 1 1 1 --openPMD.period $((10*$power)) --openPMD.file simData --openPMD.ext bp --e_macroParticlesCount.period 100

  cd ..

  dataPathCheck=$(absolute_path "$simPath/openPMD")
  dataPath="$simPath/openPMD/simData_%T.bp"
  echo "$dataPath"
fi

pathArray[$i]=$dataPath

done


#################################
## validate simulation results ##
#################################
if [ -d $dataPathCheck ] ; then
    echo "Validate!"
    ./bin/validate.sh ${pathArray[@]}
else
      echo "dataPathCheck does not exist"
      exit 3
fi
ret=$?

exit $ret
