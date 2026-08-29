#!/bin/bash
cd $SETUP_DIR
source workflow/scripts/picongpu.profile
pic-build
tbg $TBG_ARGS $RUN_DIR
