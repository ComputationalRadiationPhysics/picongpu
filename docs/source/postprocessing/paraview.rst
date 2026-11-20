.. _pp-paraview:

ParaView
========

.. sectionauthor:: Axel Huebl, Julian J. Lenz
.. moduleauthor:: Axel Huebl

`ParaView <https://www.paraview.org>`_ is an open-source, multi-platform data analysis and visualization application.
It reads PIConGPU's :ref:`openPMD output <usage-plugins-openPMD>` directly, both for interactive exploration and for
scripted, parallel post-processing.

.. note::

   The former XDMF route described in older PIConGPU documentation (ParaView 3.98, libSplash and ``pic2xdmf.py``) is
   obsolete: those tools are no longer part of PIConGPU, and current ParaView versions read openPMD files directly.

Software
--------

Use ParaView 5.9 or newer; the openPMD-api documentation currently recommends 5.11 or newer.
ParaView's openPMD reader is implemented on top of the Python bindings of
`openPMD-api <https://openpmd-api.readthedocs.io/>`_ and supports the ADIOS2 and HDF5 backends.
The ``openpmd_api`` module therefore has to be importable by the Python interpreter that ParaView uses, not only by a
separate analysis environment; otherwise ParaView reports a missing ``openpmd_api`` module.
See the `openPMD-api installation instructions <https://openpmd-api.readthedocs.io/en/latest/install/install.html>`_
for the available installation methods (PyPI, Conda, Spack, ...).
The openPMD-api installation must provide the same backend as the data, i.e. ADIOS2 for ``bp5`` output or HDF5 for
``h5`` output.

Opening a data series
---------------------

By default, PIConGPU writes one data set per iteration, e.g. ``simData_000000.bp5``, ``simData_000128.bp5``, ...
(``--openPMD.period`` and ``--openPMD.ext`` control the output interval and the backend; see
:ref:`the openPMD plugin documentation <usage-plugins-openPMD>`).
ParaView needs a small text file with the suffix ``.pmd`` that describes such a file-based series.
Its content is the file name pattern with the iteration placeholder, using the same expansion rules as the
``openPMD_api.Series`` constructor. For the PIConGPU defaults ``file = simData``, ``infix = _%06T`` and
``ext = bp5``, create the file next to the data:

.. code-block:: console

   $ cd <path-to-run>/simOutput/openPMD
   $ echo 'simData_%06T.bp5' > simData.pmd

Here ``%06T`` is the iteration number padded to six digits. For HDF5 output use the corresponding pattern, e.g.
``simData_%06T.h5``.

Open the ``simData.pmd`` file in ParaView (``File`` -> ``Open``) and select the data series.
Individual iterations, fields and particle species can then be selected in ParaView's pipeline browser, e.g. with the
``Extract Block`` filter.

.. tip::

   In ParaView's global settings (``Edit`` -> ``Settings`` on Linux), enable
   ``General`` -> ``Advanced`` -> search for ``data`` -> ``Data Processing Options`` -> ``Auto Convert Properties``.
   This simplifies the application of filters, e.g. contouring individual components of vector fields, without first
   adding a calculator that extracts a single component or the magnitude.

.. warning::

   As of ParaView 5.11 and older, axis labels are not read for fields; see `WarpX issue 1803
   <https://github.com/BLAST-WarpX/warpx/issues/1803>`_. Apply a rotation, e.g. ``0 -90 0``, to mesh data where needed.

.. warning::

   To visualize particle traces with the ``Temporal Particles To Pathlines`` filter, apply the ``Merge Blocks`` filter
   first (`ParaView issue 21837 <https://gitlab.kitware.com/paraview/paraview/-/issues/21837>`_). With multiple species,
   you may have to extract the species of interest with ``Extract Block`` before applying ``Merge Blocks``.

Working with data on an HPC system
----------------------------------

If the output is on a cluster, either copy the relevant iterations to your local machine or use ParaView's
client/server mode, so that reading and rendering happen next to the data:

* Start ``pvserver`` on a compute node, e.g. inside a batch job. It prints a connection URL of the form
  ``cs://<host>:11111``.
* In the local ParaView client, add a server under ``Connect`` with the host name and port reported by ``pvserver``,
  and connect to it. If the compute node is not directly reachable, use the SSH-tunnel support in the server
  configuration.
* Rendering can happen on the server (remote rendering, headless with ``--force-offscreen-rendering``) or on the
  client; see the ParaView chapter on `remote and parallel visualization
  <https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html>`_, the
  `SSH-tunnel server configuration
  <https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html#case-sixteen-secured-connection-to-a-server-through-ssh-tunnel>`_
  and the `pvserver command-line options
  <https://docs.paraview.org/en/latest/UsersGuide/commandLineArguments.html#pvserver-specific-options>`_.

.. note::

   Older PIConGPU instructions contained HZDR/Hypnos-specific batch scripts and a ``hypnos.pvsc`` server
   configuration. Those files and the referenced ParaView 4.x module are no longer maintained; configure the server
   connection in ParaView itself as described above.

.. seealso::

   The openPMD-api documentation on `3D visualization with ParaView
   <https://openpmd-api.readthedocs.io/en/latest/analysis/paraview.html>`_ is the primary reference for the reader
   and its current limitations.
