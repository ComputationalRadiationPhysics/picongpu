Synchrotron_plugin: Testing the Synchrotron Radiation Extension
===============================================================

.. sectionauthor:: Julian Lenz

The test simulates a homogeneous electron bunch with 40 macro-particles per
cell. The electrons are initialized with a relativistic drift (initially
:math:`\gamma = 1000`) and are neither pushed nor current-deposited, so they
stay in place while the synchrotron extension emits photons in the direction
of their momentum. The background electric and magnetic fields are defined in
``fieldBackground.param``.

Before the run, ``lib/change_parameters_randomly.py`` draws a Lorentz factor
and an effective field strength from predefined ranges, derives a time step
that keeps the photon-emission probability low, and stores the chosen values
in ``simOutput/params.txt``. ``bin/ci.sh`` then builds and runs the 24x24x24
setup for 4000 steps on one GPU and writes one openPMD output.

``lib/validate.py`` compares the simulated photon energy distribution with the
analytical synchrotron probability. The test passes if the mean relative error
is at most 10 % and enough photons and histogram bins are available.

Usage
-----

.. code-block:: bash

   pic-create $PICSRC/share/picongpu/tests/Synchrotron_plugin $HOME/picInputs/synchrotron
   cd $HOME/picInputs/synchrotron
   python3 -m pip install -r lib/requirements.txt
   ./bin/ci.sh

The simulation output can be validated separately with
``./bin/validate.sh -d <path-to-simOutput>``.
