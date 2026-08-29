#!/bin/bash
cd $RUN_DIR

# either provide the necessary input definitions on the commandline
# (e.g. `build.cwl` requires at least `include_directory` and `script`):
cwltool $CWL_ARGS $SETUP_DIR/workflow/steps/build.cwl --include_directory $SETUP_DIR/include --script $SETUP_DIR/workflow/scripts/build.sh

# or write a custom `my_input.yaml` file with content like:
# include_directory: <SETUP_DIR>
# script: <SETUP_DIR>/workflow/scripts/build.sh
cwltool $CWL_ARGS $SETUP_DIR/workflow/steps/build.cwl my_input.yaml
