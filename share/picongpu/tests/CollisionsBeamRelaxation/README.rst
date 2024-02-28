Collisions: Testing example beam relaxation
===========================================

A population of electrons with a very small temperature and a drift velocity (the beam) is colliding with ions.
Due to the collisions the velocity distribution of electrons is changing and the drift momentum is transferred into the electron transversal momentum and partially into ion momenta.
In this test only the inter-collisions (between ions and electrons) are enabled.

:ref:`More details  <model-binaryCollisions::section::tests::beamRelaxation>`

Example script for submitting all tests in parallel
.. code-block:: bash
    #!/bin/bash

    # compile and submit all setups (in parallel)
    BASE_OUTPUT_PATH=<output path> # change it
    WORKING_DIR=$(pwd)
    INPUT_PATH=$(mktemp -d -p ~/tmp/) # adjust if needed
    for i in {0..5}
    do
       cd $WORKING_DIR
       EXAMPLE_PATH=$INPUT_PATH/CASE_N$i
       mkdir $EXAMPLE_PATH
       cp -r ./ $EXAMPLE_PATH
       cd $EXAMPLE_PATH
       { pic-build -t $i &> compile.log && tbg -s -t -c etc/picongpu/1.cfg $BASE_OUTPUT_PATH/BR_CASE_N$i; } &
    done
    wait

Commands for plotting results in parallel:
.. code-block:: console
    cd <output path>
    for i in {0..5}; do python BR_CASE_N${i}/input/lib/python/picongpu/beam_relaxation_verifier/generate_plots.py --n_cells 96 --smilei_dir <path_to_smilei_results>/CASE_N${i} --file BR_main_${i} --file_debug BR_debug_${i} BR_CASE_N${i}/ & done;
    wait
