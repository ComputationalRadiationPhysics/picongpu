KelvinHelmholtz: Kelvin-Helmholtz Instability
=============================================

.. sectionauthor:: Richard Pausch <r.pausch (at) hzdr.de>
.. moduleauthor:: Richard Pausch <r.pausch (at) hzdr.de>, Mika Soren Voß <m.voss (at) hzdr.de>

This KHI growth rate test simulates a shear-flow instability known as the Kelvin-Helmholtz Instability in a sub-relativistic setup as studied in [Alves12]_, [Grismayer13]_, [Bussmann13]_. The setup uses a pre-ionized quasi-neutral hydrogen plasma. From the simulation, the growth of the dominant magnetic field is extracted and compared to theoretical predictions. The test only passes if the simulation reaches the ideal theoretical growth rate with a 10% accuracy.


Usage
-----

.. code-block:: bash

  # execute ci
  $PICSRC/share/picongpu/tests/KHI_growthRate/bin/ci.sh $PICSRC/share/picongpu/tests/KHI_growthRate/ <user_given_output_directory>


Please note that the test requires the environment variable ``$PICSRC``, which points to the PIConGPU source code.

References
----------

.. [Alves12]
       E.P. Alves, T. Grismayer, S.F. Martins, F. Fiuza, R.A. Fonseca, L.O. Silva.
       *Large-scale magnetic field generation via the kinetic kelvin-helmholtz instability in unmagnetized scenarios*,
       The Astrophysical Journal Letters (2012),
       https://dx.doi.org/10.1088/2041-8205/746/2/L14

.. [Grismayer13]
       T. Grismayer, E.P. Alves, R.A. Fonseca, L.O. Silva.
       *dc-Magnetic-Field Generation in Unmagnetized Shear Flows*,
       Physical Reveview Letters (2013),
       https://doi.org/10.1103/PhysRevLett.111.015005

.. [Bussmann13]
       M. Bussmann, H. Burau, T.E. Cowan, A. Debus, A. Huebl, G. Juckeland, T. Kluge, W.E. Nagel, R. Pausch, F. Schmitt, U. Schramm, J. Schuchart, R. Widera.
       *Radiative Signatures of the Relativistic Kelvin-Helmholtz Instability*,
       Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (2013),
       http://doi.acm.org/10.1145/2503210.2504564
                
