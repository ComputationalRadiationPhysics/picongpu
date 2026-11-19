.. _usage-python-utils:

Python Utilities
================

.. sectionauthor:: Julian J. Lenz

The ``picongpu`` Python package bundles our :ref:`PICMI <python_package/quickstart:Quick Start>` interface, the plugin output processing tools described in :ref:`Python Postprocessing <development-pytools>`, and the standalone utilities listed on this page.
We suggest installing the package with pip.
Due to how our current PICMI implementation manages relative paths it has to be installed with the ``-e`` option (in edit mode, without copying source files out of the PIConGPU source).
This will also install all the required dependencies and the console scripts.

.. code:: shell

   pip install -e $PICSRC/lib/python/

.. note::

   For a pure analysis environment (e.g. on a laptop, without compiling or running PIConGPU), a community-maintained
   `Micromamba environment file <https://gist.github.com/steindev/c1d678375fd0f974f4b9cb1480b3da03>`_ is an alternative to
   the pip install above; install `Micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_
   and follow the instructions in the file.

Overview
--------

.. list-table::
   :header-rows: 1

   * - Utility
     - Access
     - Purpose
   * - ``pic-deps``
     - ``pic-deps install`` / ``pic-deps check``
     - Install or verify the compile-time dependencies of a preset (see :ref:`usage-python-utils-console-scripts`).
   * - ``picrc-builder``
     - ``picrc-builder [config]``
     - Interactively create or complete a ``.picongpurc.toml`` (see :ref:`usage-python-utils-console-scripts`).
   * - ``MemoryCalculator``
     - ``from picongpu.extra.utils import MemoryCalculator``
     - Estimate the memory requirement of a simulation per device (see :ref:`usage-python-utils-memory-calculator`).
   * - ``FindTime``
     - ``from picongpu.extra.utils import FindTime``
     - Convert between PIConGPU iterations and simulation times (see :ref:`usage-python-utils-find-time`).
   * - ``FieldIonization``
     - ``from picongpu.extra.utils import FieldIonization``
     - Reference field-ionization rates and atomic units (see :ref:`usage-python-utils-field-ionization`).
   * - ``param_parser``
     - ``from picongpu.extra.utils import param_parser``
     - Turn JSON parameter-scan files into ``pic-configure``/``tbg`` arguments (see :ref:`usage-python-utils-param-parser`).
   * - ``FLYonPICRateCalculationReference``
     - ``from picongpu.extra.utils import FLYonPICRateCalculationReference``
     - Reference rate and cross-section implementation for FLYonPIC (see :ref:`usage-python-utils-fly-on-pic-rates`).
   * - ``plot_mpl`` visualizers
     - ``from picongpu.extra.plugins.plot_mpl import EnergyHistogramMPL``
     - Matplotlib plots of plugin output (see :ref:`usage-python-utils-plugins`).

.. toctree::
   :maxdepth: 2

   python_utils/console_scripts
   python_utils/memory_calculator
   python_utils/find_time
   python_utils/field_ionization
   python_utils/param_parser
   python_utils/fly_on_pic_rates
   python_utils/plugins

.. note::

   The shell workflow tools ``pic-build``, ``pic-configure``, ``pic-create``, ``pic-edit`` and ``tbg`` are part of the C++ workflow, not Python utilities.
   They are documented in :doc:`Basics <basics>` and :ref:`usage-tbg`.
