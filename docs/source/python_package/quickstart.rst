Quick Start
===========

.. sectionauthor:: Julian J. Lenz

This section gets you from zero to a running PIConGPU simulation
in five steps.
For everything beyond this minimal workflow,
follow the references to the other chapters.

Step 1: Install ``uv``
----------------------

.. literalinclude:: snippets/running_simulation/uv_install.sh
   :language: bash
   :start-after: BEGIN-UV-INSTALL
   :end-before: END-UV-INSTALL

``uv`` is a fast Python package installer and runner
(see :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
for alternatives and details).

Step 2: Build Your Runtime Configuration
----------------------------------------

.. literalinclude:: snippets/quickstart/picrc_builder.sh
   :language: bash
   :start-after: BEGIN-PICRC-BUILDER
   :end-before: END-PICRC-BUILDER

``picrc-builder`` guides you interactively through writing the
``.picongpurc.toml`` runtime configuration.
See :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.

Step 3: Install the Dependencies
--------------------------------

.. literalinclude:: snippets/quickstart/pic_deps_install.sh
   :language: bash
   :start-after: BEGIN-PIC-DEPS-INSTALL
   :end-before: END-PIC-DEPS-INSTALL

``pic-deps`` is a best-effort tool to system-specific installation of dependencies and might not be available for all presets.
Details on manual dependency installation can be found in the
:ref:`Dependencies <install/dependencies:Dependencies>` chapter.

Step 4: Download the Minimal Example
------------------------------------

.. literalinclude:: snippets/quickstart/download_minimal_example.sh
   :language: bash
   :start-after: BEGIN-DOWNLOAD-MINIMAL-EXAMPLE
   :end-before: END-DOWNLOAD-MINIMAL-EXAMPLE

This is the minimal input script defining an empty void to simulate.
See :ref:`Defining Your Simulation <python_package/foundations/defining_simulation:Defining Your Simulation>`
to add particles, lasers and diagnostics to this skeleton.

.. dropdown:: The full minimal example

   The downloaded file defines a ``Simulation`` with a ``max_steps`` budget
   and an electromagnetic solver on a 3D grid, then calls ``simulation.run()``
   to generate the input files, compile a tailored binary and submit the run.

   .. literalinclude:: ../../../lib/python/examples/tutorial/01_minimal.py
      :language: python

Step 5: Run It
--------------

.. literalinclude:: snippets/quickstart/run_minimal_example.sh
   :language: bash
   :start-after: BEGIN-RUN-MINIMAL-EXAMPLE
   :end-before: END-RUN-MINIMAL-EXAMPLE

The script carries `PEP 723 inline script metadata <https://peps.python.org/pep-0723/>`__,
so ``uv`` installs PIConGPU.
See :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
for what happens under the hood and where the results end up.

Next Steps
----------

* The :ref:`tutorial <python_package/tutorial:Tutorial: Setting up a simple LWFA>`
  walks through a complete laser-wakefield-acceleration example.
* The core concepts and configuration options are introduced in
  :ref:`Foundations <python_package/foundations/index:Foundations>`.
* Particular features
  (lasers, species, distributions, interactions, diagnostics, ...)
  are covered in :ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`.
* The complete reference of the PICMI frontend is the
  :ref:`API Documentation <python_package/api/index:API Documentation>`.
