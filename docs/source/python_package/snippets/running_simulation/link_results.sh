#!/bin/bash
# general linking to a user-defined location
$RUN_DIR/link_results.sh /path/to/my/results

# default linking, restoring the behaviour as found in the legacy workflow
cd $RUN_DIR
./link_results.sh
