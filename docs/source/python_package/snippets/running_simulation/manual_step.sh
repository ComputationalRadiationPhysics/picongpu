#!/bin/bash
# BEGIN-MANUAL-STEP
mkdir $RUN_DIR/build_step
cd $RUN_DIR/build_step
ln -s $SETUP_DIR/include
ln -s $SETUP_DIR/workflow/scripts/build.sh
./build.sh
# END-MANUAL-STEP
