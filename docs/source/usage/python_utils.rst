.. SPDX-FileCopyrightText: PIConGPU contributors
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _usage-python-utils:

Python Utilities
================

PIConGPU python package includes plugin output processing tools, tools listed below and our :ref:`PICMI <PICMI>` and PyPIConGPU interface.
We suggest installing the picongpu package with pip.
Due to how our current PICMI implementation manages relative paths it has to be installed with the `-e` option (in edit mode, without copying source files out of PIConGPU source).
This will also install all the required dependencies.

.. code:: shell

   pip install -e $PICSRC/lib/python/

This section contains python utilities for more comfortable working with PIConGPU.

.. toctree::
   :maxdepth: 2

   python_utils/memory_calculator
