.. _development-index:

Development
===========

.. sectionauthor:: Julian J. Lenz

This chapter is for people who contribute to PIConGPU itself: it covers the
contribution workflow and continuous integration, coding style and developer
tooling, the C++ core and the Python tooling, the test suite, and the sources
of this manual.

.. TODO (human): replace the factual paragraph above with the orientation
   paragraph for the three audiences (first-time contributor, C++/Python core
   developer, documentation editor).

.. toctree::
   :caption: Contributing
   :maxdepth: 1

   CONTRIBUTING.md
   docs/COMMIT.md
   repostructure
   ci

.. toctree::
   :caption: Style & tooling
   :maxdepth: 1

   styleguide
   debugging

.. toctree::
   :caption: C++ core
   :maxdepth: 1

   extending
   ../prgpatterns/lockstep
   picongpu
   pmacc

.. toctree::
   :caption: Python tooling
   :maxdepth: 1

   py_postprocessing

.. toctree::
   :caption: Testing
   :maxdepth: 1

   ../testing/index

.. toctree::
   :caption: Documentation
   :maxdepth: 1

   sphinx
   PlantUML
