atomicPhysics: atomic Physics example for experimental FLYonPIC PIConGPU extension
    was originally based on the Empty example by Axel Huebl
============================

.. sectionauthor:: Brian Marre <b.marre (at) hzdr.de>
This is a minimum spec-exmaple for testing the still experimental atomic physics branch of
picongpu.
It uses mostly default picongpu algorithms and settings, except for a few
changes listed below, to reduce computation time.
 - reduced super cell size, only 2x2x2 cells form a super cell
 - reduced number of particels overall, only 1 macro-ion and 1 macro-electron per super cell
 - no ionization, instead initialization in a somewhat correct charge state for initial
    conidtions

Use this as a starting point for your own atomic physics picongpu simulations, but beware
this is still experimental
