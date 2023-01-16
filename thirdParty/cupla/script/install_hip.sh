#!/bin/bash

if ! [ -z ${CI_GPUS+x} ] && [ -n "$CI_GPUS" ] ; then
    # select randomly a device if multiple exists
    # CI_GPUS is provided by the gitlab CI runner
    SELECTED_DEVICE_ID=$((RANDOM%CI_GPUS))
    export HIP_VISIBLE_DEVICES=$SELECTED_DEVICE_ID
    export CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE_ID
    echo "selected device '$SELECTED_DEVICE_ID' of '$CI_GPUS'"
else
    echo "No GPU device selected because environment variable CI_GPUS is not set."
fi

if [ -z ${CI_GPU_ARCH+x} ] ; then
    # In case the runner is not providing a GPU architecture e.g. a CPU runner set the architecture
    # to Radeon VII or MI50/60.
    export GPU_TARGETS="gfx906"
fi
