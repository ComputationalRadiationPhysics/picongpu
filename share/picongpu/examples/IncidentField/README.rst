IncidentField: Laser Injection with the Incident-Field Solver
=============================================================

.. sectionauthor:: Julian J. Lenz

This field-only example demonstrates laser injection through PIConGPU's
incident-field (total-field/scattered-field) source. Two Gaussian pulses are
launched into the box: one from the XMin boundary with :math:`E_y` and
:math:`B_z`, and one from the YMax boundary with :math:`E_x` and :math:`B_z`.
The YMax profile is used twice, once directly and once delayed by 100 time
steps, to show that several profiles can be superposed on one boundary.

The profiles are implemented as ``profiles::Free`` functors in
``incidentField.param``. Their common parameters are a wave period of
0.0211 ns (wavelength of about 6.3 mm), :math:`a_0 = 8`, a duration of
0.026 ns, and Gaussian transverse envelopes with widths of 8 mm in x and y and
10 mm in z. The box has 256 cells per direction at 0.5 mm cell size, uses the
Yee solver and runs for 1000 steps. There are no particles in this setup.

``1.cfg`` runs on one GPU and ``4.cfg`` on four GPUs. ``cmakeFlags`` also
provides a 2D configuration. PNG snapshots of the fields are written every 10
steps until step 650, openPMD-HDF5 output every 50 steps, and the field energy
every 10 steps.

Usage
-----

.. code-block:: bash

   pic-create $PIC_EXAMPLES/IncidentField $HOME/picInputs/incidentField
   cd $HOME/picInputs/incidentField
   pic-build
   tbg -s bash -c etc/picongpu/1.cfg -t $PICSRC/etc/picongpu/bash/mpiexec.tpl $SCRATCH/runs/incidentField
