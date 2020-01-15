atomicPhysics: atomic Physics example for PIConGPU based on the Empty example by Axel Huebl
============================

.. sectionauthor:: Axel Huebl <a.huebl (at) hzdr.de> and Brian Marre <b.marre (at) hzdr.de>

original Example Description:
This is an "empty" example, initializing a default particle-in-cell cycle with default algorithms [BirdsallLangdon]_ [HockneyEastwood]_ but without a specific test case.
When run, it iterates a particle-in-cell algorithm on a vacuum without particles or electro-magnetic fields initialized, which are the default ``.param`` files in ``include/picongpu/param/``.

This is a case to demonstrate and test these defaults are still (syntactically) working.
In order to set up your own simulation, there is no need to overwrite all ``.param`` files but only the ones that are different from the defaults.
As an example, just overwrite the default laser (none) and initialize a species with a density distribution.

changes made:
As an example I added the new particle attributes of topic-atomicPhysics
to the species definitions in speciesDefinition.param as well as define the
necessary particle attributes in the default speciesAttributes.param file.


References
----------

.. [BirdsallLangdon]
        C.K. Birdsall, A.B. Langdon.
        *Plasma Physics via Computer Simulation*,
        McGraw-Hill (1985),
        ISBN 0-07-005371-5

.. [HockneyEastwood]
        R.W. Hockney, J.W. Eastwood.
        *Computer Simulation Using Particles*,
        CRC Press (1988),
        ISBN 0-85274-392-0
