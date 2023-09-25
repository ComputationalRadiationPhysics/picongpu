.. _development-repostructure:

Repository Structure
====================

.. sectionauthor:: Axel Huebl

Branches
--------

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

      * python modules
      * set ``PYTHONPATH`` here
      * ``extra/``

        * modules, e.g. for RT interfaces, pre* & post-processing

      * ``picmi/``

        * user-facing python interface

      * ``pypicongpu/``

        * internal interface for ``.param`` & ``.cfg``-file generation
        * used by PICMI implementation

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
    * ``pypicongpu/``: required files for code generation

      * ``schema/``: code generation JSON schemas
      * ``template/``: base template for code generation

* ``bin/``

  * core tools for the "PIConGPU framework"
  * set ``PATH`` here

* ``docs/``

  * currently for the documentation files
  * might move, e.g. to ``lib/picongpu/docs/`` and its build artifacts to ``share/{doc,man}/``, 
