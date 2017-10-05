.. _development-repostructure:

Repository Structure
====================

.. sectionauthor:: Axel Huebl

Branches
--------

* ``master``: the latest stable release, always tagged with a version
* ``dev``: the development branch where all features start from and are merged to
* ``release-X.Y.Z``: release candiate for version ``X.Y.Z`` with an upcoming release, receives updates for bug fixes and documentation such as change logs but usually no new features

Directory Structure
-------------------

* ``include/``

  * C++ header *and* source files
  * set ``-I`` here
  * prefixed with project name

* ``lib/``

    * pre-compiled libraries
    * ``python/``

      * modules, e.g. for RT interfaces, pre* & post-processing
      * set ``PYTHONPATH`` here

* ``etc/``

  * (runtime) configuration files
  * ``picongpu/``

    * ``tbg`` templates (as long as PIConGPU specific, later on to ``share/tbg/``)
    * network configurations (e.g. infiniband)
    * score-p and vampir-trace filters

* ``share/``

  * examples, documentation
  * ``picongpu/``

    * ``completions/``: bash completions
    * ``examples/``: each with same structure as ``/``

* ``bin/``

  * core tools for the "PIConGPU framework"
  * set ``PATH`` here

* ``docs/``

  * currently for the documentation files
  * might move, e.g. to ``lib/picongpu/docs/`` and its build artifacts to ``share/{doc,man}/``, 
