#!/bin/bash
# BEGIN-CWLTOOL-WORKFLOW
CWL_ARGS="--leave-tmpdir --preserve-entire-environment --cachedir=.cwl_cache"
cd $RUN_DIR
cwltool $CWL_ARGS $SETUP_DIR/workflow/workflow.cwl $SETUP_DIR/workflow/input.yaml
./link_results.sh
# END-CWLTOOL-WORKFLOW
