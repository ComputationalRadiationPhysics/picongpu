Intro
=====

This is a brief overview over PICMI.

Example
-------

.. literalinclude:: ../../../../share/picongpu/pypicongpu/examples/warm_plasma/main.py
   :language: python

Creates a directory ``generated_input``, where you can run ``pic-build`` and subsequently ``tbg``.

.. note::

   Note that in order to run the python script, you need to set up a Python environment that
   includes all the dependencies listed in  ``$PICSRC/lib/python/picongpu/picmi/requirements.txt``.
   This can be done by e.g. ``pip install -r $PICSRC/lib/python/picongpu/picmi/requirements.txt``.
   You will also need to set your ``$PYTHONPATH`` to include PIConGPU's Python libraries.
   This can be done by ``export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH``.
   (Updating the ``PYTHONPATH`` is automatically done when you source your PIConGPU environment, see below.)
   For ``pic-build`` and ``tbg`` you need to setup a working PIConGPU environment.
   This is documented in :ref:`the Setup part of PIConGPU in 5 Minutes on Hemera <hemeraIn5min>`.

   Above, we used:

   -  ``$PICSRC`` as shell variable providing the path to picongpu's source code directory.


Generation Results
------------------

The recommended way to use the generated simulations is to

1. create the simulation in PICMI
2. call ``Simulation.write_input_file(DIR)``
3. use the normal PIConGPU toolchain (``pic-build``, ``tbg``)

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

Output
------

.. warning::

   This is subject to change.

Output is currently **not configurable**.

Some output is automatically enabled, including PNGs.
For this the period is chosen that the output is generated (approx.) 100 times over the entire simulation duration.

To configure output you must :ref:`change the generated files <picmi-custom-generation>`.

Unsupported Features
--------------------

You will be alerted for unsupported features.

Either a warning is printed or an error thrown (including because a class does not exist).
In this case read the error message to fix this.

For reference you can see how the tests in ``$PICSRC/test/python/picongpu/quick/picmi`` use the interface.

Reference
---------

The full PICMI reference is available `upstream <https://picmi-standard.github.io/>`_.

Extensions
----------

Parameters/Methods prefixed with ``picongpu_`` are PIConGPU-exclusive.

.. warning::

   We strive to quickyl contribute these parameters to PICMI upstream,
   so this list is to be considered volatile.

- Simulation

  - ``picongpu_get_runner()``:
    Retrieve a :ref:`PyPIConGPU Runner <pypicongpu-running>`
  - ``picongpu_template_dir``:
    Specify the template dir to use for code generation,
    please refer to :ref:`the documentation on the matter for details <picmi-custom-generation>`

- Grid
  
  - ``picongpu_n_gpus``:
    list of a 1 or 3 integers, greater than zero, describing GPU distribution in space
    3-integer list: ``[N_gpu_x, N_gpu_y, N_gpu_z]``
    1-integer list: ``[1, N_gpu_y, 1]``
    Default is ``None`` equal to ``[1, 1, 1]``

- Gaussian Laser

  - Laguerre Modes (``picongpu_laguerre_modes`` and ``picongpu_laguerre_phases``):
    Two lists of float, passed to PIConGPU laser definition

- Species

  - ``picongpu_ionization_electrons``:
    Electron species to use for ionization.
    Optional, will be guessed if possible.
  - ``picongpu_fully_ionized``:
    When defining an element (using ``particle_type``) it may or may not be ionizable

    - to **enable** ionization simulation set ``charge_state`` to an integer
    - to **disable** ionization (ions are only core without electrons) set ``picongpu_fully_ionized=True``

    If neither is set a warning is printed prompting for either of the options above.
