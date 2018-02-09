Empty: Default PIC Algorithm
============================

.. sectionauthor:: Axel Huebl <a.huebl (at) hzdr.de>

This is an "empty" example, initializing a default particle-in-cell cycle with default algorithms [BirdsallLangdon]_ [HockneyEastwood]_ but without a specific test case.
When run, it iterates a particle-in-cell algorithm on a vacuum without particles or electro-magnetic fields initialized, which are the default `.param` files in `include/picongpu/param/`.

This is a case to demonstrate and test these defaults are still (syntactically) working.
In order to set up your own simulation, there is no need to overwrite all `.param` files but only the ones that are different from the defaults.
As an example, just overwrite the default laser (none) and initialize a species with a density distribution.


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
