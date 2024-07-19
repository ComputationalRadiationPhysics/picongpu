.. _PICMI:
Intro
=====

PICMI is a Python interface for configuring and running Particle-In-Cell simulations, defined via the Python reference implementation
available from `github <https://github.com/picmi-standard/picmi>`_ and `pypip <https://pypi.org/project/picmistandard/>`_.

PICMI allows a user to configure a simulation in a Python script, called *user script*, by building a simulation Python object from different
standardised building blocks.

.. figure::  ../media/PICMI_structure_DetailMiddle.png
   :name: PICMI_mediumDetail
   :width: 100.0%

   general overview of PICMI interface, see the `reference implementation <https://github.com/picmi-standard/picmi>`_ for details

From this simulation object a user may then generate input files for all PIC-simulation codes supporting PICMI, but some features may be code specific or not supported by all codes.

.. warning::

   It is highly discouraged to edit generated PIConGPU input files **after** generation.

  It is very easy to make mistakes this way, use the process outlined in :ref:`<picmi-custom-generation>` for customizing the generation of PIConGPU setups with PICMI.

Usage Quick-start
-----------------

To use the PICMI interface you need a working PIConGPU environment, see :ref:`install instructions <install-path>` and :ref:`the Setup part of PIConGPU in 5 Minutes on Hemera <hemeraIn5min>` for instructions.

In addition you need to install the Python dependencies of the PIConGPU PICMI implementation to your Python environment.

To install the Python dependencies you may either run the command below

.. code:: shell

  pip install -r $PIC_SRC/lib/python/piconpgu/picmi/requirements.txt

or install all the requirements listed in

- ``$PIC_SRC/lib/python/piconpgu/picmi/requirements.txt``
- and ``$PIC_SRC/lib/python/piconpgu/pypicongpu/requirements.txt``

After you have installed the dependencies you must include the PIConGPU PICMI implementation in your ``PYHTONPATH`` environment variable, for example by

.. code:: shell

  export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH

.. note::

  If you are using one of our pre-configured profiles, this is done automatically when you source a profile

.. note::
   Above, we used ``$PICSRC`` as a short hand for the path to picongpu's source code directory, provided from your shell environment if a pre-configured profile is used.

After you have installed all PICMI dependencies, simply create a user script, see :ref:`here <example_PICMI_setup>`, and generate a picongpu setup, see :ref:`generating a PIConGPU setup with PICMI <generating_setups_with_PICMI>`.

Example User Script for a warm plasma setup:
--------------------------------------
.. _example_PICMI_setup:

.. literalinclude:: ../../../../share/picongpu/pypicongpu/examples/warm_plasma/main.py
   :language: python

Creates a directory ``generated_input``, where you can run ``pic-build`` and subsequently ``tbg``.

Generation of PIConGPU setups with PICMI
----------------------------------------
.. _generating_setups_with_PICMI:

The recommended way to use the generated simulations is to

1. create the simulation in the PICMI
2. call ``simulation.write_input_file(DIR)``
3. use the normal PIConGPU toolchain (``pic-build``, ``tbg``) on the generated PIConGPU setup

.. note::

   Rationale: PICMI does not (yet) support enough parameters to meaningfully control the execution process.

Additionally, the following methods work (but are **not recommended**):

- call ``Simulation.step(NUM)``

  - directly builds and runs the simulation
  - ``NUM`` must be the **maximum number of steps**
  - has no diagnostic output (i.e. console hangs without output)

- call ``Simulation.picongpu_run()``

  - equivalent to ``Simulation.step()`` with the maximum number of steps

- use the :ref:`PyPIConGPU runner <pypicongpu-running>`

PICMI Reference
---------------

The full PICMI standard interface reference is available `upstream <https://picmi-standard.github.io/>`_.

PIConGPU specifics
------------------

PIConGPU has it's own implementation of the PICMI standard with picongpu specific extensions.
Therefore all PICMI implementations for use with PIConGPU are located in the ``picongpu.pimci`` namespace instead of the usual ``picmistandard`` namespace.

In addition names of classes differ a little from the standard names, specifically we usually strip the `PICMI_` prefix, and sometimes a class may provide additional options, see :ref:`Extensions <_PICMI_Extensions>` below.

Extensions
^^^^^^^^^^
.. _PICMI_Extensions:

Parameters/Methods prefixed with ``picongpu_`` are PIConGPU-exclusive.

.. warning::

   We strive to quickly contribute these parameters to PICMI upstream,
   so this list is to be considered volatile.

- **Simulation**

  - ``__init__(..., picongpu_template_dir)``:
    Specify the template dir to use for code generation,
    please refer to :ref:`the documentation on the matter for details <picmi-custom-generation>`
  - ``__init__(...,  picongpu_custom_user_input)``:
    pass custom user input to the code generation. this may be used in conjunction with custom templates to change the code generation.
    See :ref:`PICMI custom code generation<picmi-custom-generation>` for the documentation on using custom input.
  - ``__init__(...,  picongpu_typical_ppc)`` typical ppc to be used for normalization in PIConGPU
  - ``write_input_file(..., pypicongpu_simulation)``:
    use a :ref:`PyPIConGPU simulation<PyPIConGPU_Intro>` object instead of an PICMI- simulation object to generate a PIConGPU input.
  - ``get_as_pypicongpu()``:
    convert the PICMI simulation object to an equivalent :ref:`PyPIConGPU <PyPIConGPU_Intro>` simulation object.
  - ``picongpu_get_runner()``:
    Retrieve a :ref:`PyPIConGPU Runner <pypicongpu-running>` for running a PIConGPU simulation from Python, **not recommended**
  - ``picongpu_add_custom_user_input()``:
    add custom user input to the simulation

 - **Grid**

  - ``picongpu_n_gpus``:
    list of a 1 or 3 integers, greater than zero, describing GPU distribution in space
    3-integer list: ``[N_gpu_x, N_gpu_y, N_gpu_z]``
    1-integer list: ``[1, N_gpu_y, 1]``
    Default is ``None`` equal to ``[1, 1, 1]``

- **Gaussian Laser**

  - Laguerre Modes (``picongpu_laguerre_modes`` and ``picongpu_laguerre_phases``):
    Two lists of float, passed to PIConGPU laser definition to use laguerre modes for laser description
  - ``picongpu_polarization_type`` configuration of polarization of the laser, either linear or circular, default is linear.
  - ``picongpu_phase`` phase offset of the laser
  - ``picongpu_huygens_surface_positions`` configuration of the position of the hygens surface used by PIConGPU for laser feed in


- **Species**

  - ``picongpu_ionization_electrons``:
    Electron species to use for ionization.
    Optional, will be guessed if possible.
  - ``picongpu_fully_ionized``:
    When defining an element (using ``particle_type``) it may or may not be ionizable

    - to **enable** ionization simulation set ``charge_state`` to an integer
    - to **disable** ionization (ions are only core without electrons) set ``picongpu_fully_ionized=True``

    If neither is set a warning is printed prompting for either of the options above.

Output
^^^^^^
Output is currently **not configurable** for picongpu using the PICMI interface.

.. warning::

   This is subject to change.

If you are using the the default templates some output is automatically enabled, including PNGs.
For this the period is chosen that the output is generated (approx.) 100 times over the entire simulation duration.

To configure output you must :ref:`change the generated files <picmi-custom-generation>` by using use custom user input and custom templates.

Unsupported Features
^^^^^^^^^^^^^^^^^^^^

The PIConGPU PICMI interface currently does not support the entire PICMI interface due to standard ambiguities and/or incomplete implementation.
If you try to use an unsupported feature, you will be alerted by either a warning printed to ``stdout`` or an error thrown (including because a class does not exist).

In this case read the error message to fix this.

For reference you can see how the tests in ``$PICSRC/test/python/picongpu/quick/picmi`` use the interface.

.. note::

  If a feature is not out of the box supported by the PIConGPU PICMI interface but supported by PIConGPU it may still be configured in PICMI through :ref:`custom code generation <picmi-custom-generation>`.

  **In general everything that may be configured using the extensive PIConGPU** ``.param files`` **, may by configured via PICMI using custom input and custom templates.**

.. note::

  Please consider contributing all custom template features and custom user input back to PIConGPU to allow further improvements of the standard, making your life and everybody else' easier.

PyPIConGPU
----------

.. _PyPIConGPU_Intro:

In addition to the PICMI interface PIConGPU has a second Python interface called PyPIConGPU.
This is the native Python interface of PIConGPU to which PICMI inputs are converted to.

This interface offers additional configuration options above and beyond the PICMI interface and may be used instead of PICMI to configure a PIConGPU simulation.

.. note::

  To generate a PIConGPU setup from a PyPIConGPU simulation object use the following

  .. code:: python
    import picongpu

    picongpu.picmi.Simulation().write_input_file(<path of setup to generate>, pypicongpu_simulation)

.. note::

  The PyPIConGPU interface may be used stand-alone or together with the PICMI interface.

  In the latter case configure the PICMI simulation first, generate a PyPIConGPU simulation from the PICMI simulation and then continue configuring the PyPIConGPU object.

  **Changes of the PICMI object after creation of the PyPIConGPU simulation will not be reflected in the PyPIConGPU simulation object.**
  Both simulations representations are independent of each other, after generation.
