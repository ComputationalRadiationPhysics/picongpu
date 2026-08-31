#!/bin/bash
# BEGIN-LEGACY-WORKFLOW
cd $SETUP_DIR
source workflow/scripts/picongpu.profile
pic-build
tbg $TBG_ARGS $RUN_DIR
# END-LEGACY-WORKFLOW
