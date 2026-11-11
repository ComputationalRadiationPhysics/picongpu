Installing the Python Package
=============================

.. sectionauthor:: Julian J. Lenz

PIConGPU can be conveniently installed as a Python package which bundles the
C++ core components, a Python interface and some convenience tooling.
We list the most commonly used methods below, all of which currently fetch the
code from our `GitHub source repository
<https://github.com/ComputationalRadiationPhysics/picongpu>`__.
We strongly recommend to replace the ``@dev`` with a specific version tag or
commit hash in order to pin a version.

The Python tooling is built to handle runtime dependencies smoothly.
In some edge cases, you might need to install some dependencies manually as
described :ref:`here <install/dependencies:Dependencies>`.

With ``pip``
------------

Install into a `virtual environment
<https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`__
(e.g. via `venv <https://docs.python.org/3/library/venv.html>`__,
`uv <https://docs.astral.sh/uv/>`__, `mamba <https://mamba.readthedocs.io/>`__):

.. literalinclude:: ../snippets/running_simulation/pip_install_from_git.sh
   :language: bash
   :start-after: BEGIN-PIP-INSTALL-FROM-GIT
   :end-before: END-PIP-INSTALL-FROM-GIT

With ``uv``
-----------

`uv <https://docs.astral.sh/uv/>`__ is a fast Python package installer and
runner.
Install it with

.. literalinclude:: ../snippets/running_simulation/uv_install.sh
   :language: bash
   :start-after: BEGIN-UV-INSTALL
   :end-before: END-UV-INSTALL

and install PIConGPU as an isolated command line tool with

.. literalinclude:: ../snippets/running_simulation/uv_tool_install.sh
   :language: bash
   :start-after: BEGIN-UV-TOOL-INSTALL
   :end-before: END-UV-TOOL-INSTALL

``uv tool install`` places the package's console scripts on your ``PATH``.
You can also skip the installation entirely and let ``uv run`` resolve the
package on the fly from the
`PEP 723 inline script metadata <https://peps.python.org/pep-0723/>`__ in your
input file (see the :ref:`Quick Start <python_package/quickstart:Quick Start>`).

With ``pipx``
-------------

`pipx <https://pipx.pypa.io/>`__ installs Python applications in isolated
environments.
Install it with

.. literalinclude:: ../snippets/running_simulation/pipx_install.sh
   :language: bash
   :start-after: BEGIN-PIPX-INSTALL
   :end-before: END-PIPX-INSTALL

and install PIConGPU with

.. literalinclude:: ../snippets/running_simulation/pipx_install_package.sh
   :language: bash
   :start-after: BEGIN-PIPX-INSTALL-PACKAGE
   :end-before: END-PIPX-INSTALL-PACKAGE

From a local checkout
---------------------

To develop PIConGPU itself, install the package in editable mode from a local
clone of the repository:

.. literalinclude:: ../snippets/running_simulation/pip_install_from_source.sh
   :language: bash
   :start-after: BEGIN-PIP-INSTALL-FROM-SOURCE
   :end-before: END-PIP-INSTALL-FROM-SOURCE

Use ``-e`` so that changes in your working copy take effect without
reinstalling.
The optional test and development dependencies are listed in
`lib/python/pyproject.toml
<https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/lib/python/pyproject.toml>`__.

Next Steps
----------

* :ref:`Quick Start <python_package/quickstart:Quick Start>` -- run your first
  simulation.
* :ref:`Running Your Simulation
  <python_package/foundations/running_simulation:Running Your Simulation>` --
  the full set of invocation methods, advanced and legacy workflows, and where
  the results end up.
* :ref:`Installation (HPC / from source) <install/path:Introduction>` -- the
  manual C++ dependency installation, cluster profiles and environment setup
  needed to actually build and submit PIConGPU.
