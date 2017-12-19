.. _usage-plugins-sliceFieldPrinter:

Slice Field Printer
-------------------

Outputs a 2D slice of the **electric, magnetic and/or current field** in SI units. The slice position and the field can be specified by the user.

.cfg file
^^^^^^^^^

The plugin works on **electric**, **magnetic**, and **current** fields. 
For the electric field, the prefix ``--E_slice.`` for all command line arguments is used. 
For the magnetic field, the prefix ``--B_slice.`` is used.
For the current field, the prefix ``--J_slice.`` is used.

The following table will describe the setup for the electric field. 
The same applied to the magnetic field. 
Only the prefix has to be adjusted.

======================== ============================================================================================================================================
Command line option      Description
======================== ============================================================================================================================================
``--E_slice.period``     The periodicity of the slice print out.
                         If set to a non-zero value, e.g. to ``--E_slice.period 100``, the slices are generated for every 100th simulation time step.
``--E_slice.fileName``   Name of the output file. Setting `--E_slice.fileName myName` will result in output files like ``myName_100.dat``.
``--E_slice.plane``      Defines the plane that the slice will be parallel to.
                         The plane is defined by its orthogonal axis.
                         By using ``0`` for the x-axis, ``1`` for the y-axis and ``2`` for the z-axis, all standard planes can be selected.
                         E.g. choosing the x-y-plane is done by setting the orthogonal axis to the z-axis by giving the command line argument ``--E_slice.plane 2``.
``--E_slice.slicePoint`` Defines the position of the slice on the orthogonal axis.
                         E.g. when the x-y-plane was selected, the slice position in z-direction has to be set.
                         This is done using a value between ``0.0`` and ``1.0``. E.g. by setting ``--E_slice.slicePoint 0.5``, the slice is centered.
======================== ============================================================================================================================================

This plugin **supports using multiple slices**. By setting the command line arguments multiple times, multiple slices are printed to file. 
As an example, the following command line will create two slices:

.. code:: bash

  picongpu # [...]
    --E_slice.period 100 --E_slice.fileName slice1 --E_slice.plane 2 --E_slice.slicePoint 0.5
    --E_slice.period  50 --E_slice.fileName slice2 --E_slice.plane 0 --E_slice.slicePoint 0.25

The first slice is a cut along the x-y axis. It is printed every 100th step. It cuts through the middle of the z-axis and the data is stored in files like `slice1_100.dat`.
The second slice is a cut along the y-z axis. It is printed every 50th step. It cuts through the first quarter of the x-axis and the data is stored in files like `slice2_100.dat`.

2D fields
"""""""""

In the case of 2D fields, the plugin outputs a 1D slice. Be aware that ``--E_slice.plane`` still refers to the orthogonal axis, i.e. ``--E_slice.plane 1`` outputs a line along the **x-axis** and ``--E_slice.plane 0`` along the **y-axis**.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

the local slice is permanently allocated in the type of the field (``float3_X``).

Host
""""

as on accelerator.

Output
^^^^^^

The output is stored in an ASCII file for every time step selected by ``.period`` (see *How to set it up?*).
The 2D slice is stored as lines and rows of the ASCII file.
Spaces separate rows and newlines separate lines.
Each entry is of the format ``{1.1e-1,2.2e-2,3.3e.3}`` giving each value of the vector field separately e.g. ``{E_x,E_y,E_z}``.


In order to read this data format, there is a python module in ``lib/python/picongpu/plugins/sliceFieldReader.py``.
The function ``readFieldSlices`` needs a data file (file or filename) with data from the plugin and returns the data as numpy-array of size ``(N_y, N_x, 3)``


Known Issues
^^^^^^^^^^^^

See `issue #348 <https://github.com/ComputationalRadiationPhysics/picongpu/issues/348>`_.

Should be solved with `pull request #548 <https://github.com/ComputationalRadiationPhysics/picongpu/pull/548>`_.
