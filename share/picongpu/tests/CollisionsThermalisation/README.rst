Collisions: Testing example thermalization
==========================================

In this example there are two particle populations — electrons and ions.
They are thermally initialized with different temperatures and their temperatures get closer to each other with time.
The usual PIC steps are disabled (there is no field solver and no pusher).
The thermalization happens solely due to the binary collisions.
We enable inter-collisions for ions and electrons as well as collisions between the two species.

:ref:`More details  <model-binaryCollisions::section::tests::thermalization>`

Example script for submitting all tests in parallel
.. code-block:: bash
    #!/bin/bash

    # compile and submit all setups (in parallel)
    BASE_OUTPUT_PATH=<some path> # change it
    WORKING_DIR=$(pwd)
    INPUT_PATH=$(mktemp -d -p ~/tmp/) # adjust if needed
    for i in {0..8}
    do
       cd $WORKING_DIR
       EXAMPLE_PATH=$INPUT_PATH/CASE_N$i
       mkdir $EXAMPLE_PATH
       cp -r ./ $EXAMPLE_PATH
       cd $EXAMPLE_PATH
       { pic-build -t $i &> compile.log && tbg -s -t -c etc/picongpu/1.cfg $BASE_OUTPUT_PATH/TEST_CASE_N$i; } &
    done
    wait

Commands for plotting results in parallel:
.. code-block:: console
    cd <output path>
    for i in {0..2}; do python TEST_CASE_N0/input/lib/python/picongpu/thermalization_verifier/generate_plots.py --coulomb_log 5 --n_cells 2304 --file main_${i}.png --file_debug debug_${i}.png --smilei_dir <smilei_output_path>/N0${i} TEST_CASE_N${i} & done;
    for i in {3..8}; do python TEST_CASE_N0/input/lib/python/picongpu/thermalization_verifier/generate_plots.py --n_cells 2304 --file main_${i}.png --file_debug debug_${i}.png --smilei_dir <smilei_output_path>/N0${i} TEST_CASE_N${i} & done;
    wait
