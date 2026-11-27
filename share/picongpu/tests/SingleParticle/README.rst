SingleParticle: Testing Particle Pushers in Background Fields
=============================================================

.. sectionauthor:: Julian Lenz

A single electron is initialized in cell ``(32, 5, 16)`` of a 64x64x32 grid.
The background fields are uniform: :math:`E_z = -10` MV/m and
:math:`B_z = 50` T, so the electron gyrates in the x-y plane while drifting
along +z under the electric field. Its initial drift is
:math:`\beta = 0.5` along x; one gyration takes 0.825 ps and the time step is
set to 1/50 of that period.

The setup compiles and exercises the available particle pushers, and also
checks the case without a pusher. ``cmakeFlags`` lists the tested
configurations: Boris (default), Free, Vay, Photon and Reduced
Landau-Lifshitz, each in 3D and 2D, plus the disabled-pusher runs. openPMD
output is written every step and the electron macro-particle count every 100
steps.

Usage
-----

.. code-block:: bash

   pic-create $PICSRC/share/picongpu/tests/SingleParticle $HOME/picInputs/singleParticle
   cd $HOME/picInputs/singleParticle
   pic-build
   tbg -s bash -c etc/picongpu/1.cfg -t $PICSRC/etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/singleParticle
