.. _pp-openPMD:

openPMD
=======

.. sectionauthor:: Axel Huebl
.. moduleauthor:: Axel Huebl

.. tip::

   This page describes how PIConGPU writes openPMD data and collects tools
   for reading and analyzing it.
   For the plugin's configuration options, see
   :ref:`the openPMD plugin documentation <usage-plugins-openPMD>`;
   for the PICMI diagnostics that write openPMD, see the
   :ref:`PICMI openPMD output <openpmd>` page.
   A guided walk-through is part of the
   :ref:`legacy interface tutorial <hemeraIn5min>`.

If you are new to openPMD, the `project website <https://www.openpmd.org>`_
provides an introduction.

Output
------

PIConGPU writes fields and particles via the `openPMD API
<https://github.com/openPMD/openPMD-api>`_ following the `openPMD standard
<https://github.com/openPMD/openPMD-standard>`_.
Data is written with parallel (MPI) I/O either as ADIOS2 files
(``.bp5`` by default when a recent ADIOS2 is available, ``.bp4`` for older
ADIOS2 versions, and the legacy ``.bp`` extension) or as HDF5 files
(``.h5``).
The openPMD API selects the backend from the file extension.
Streaming (in-situ) output via ADIOS2/SST is available as well.

By default, output is written to :file:`simOutput/openPMD/` as one file per
time step, with names such as :file:`simData_000000.bp5`.
The :ref:`openPMD plugin documentation <usage-plugins-openPMD>` covers all
configuration options, including the iteration encoding, the output range and
backend-specific settings.

PIConGPU requires the openPMD API in version 0.17.0 or greater;
see :ref:`the dependencies <install-dependencies>`.

Reading and analysis
--------------------

As a user of PIConGPU, you will be mainly interested in our
:ref:`python tools <usage-python-utils>` and readers, that can read openPMD, e.g. into:

* read & write data: `openPMD-api <https://github.com/openPMD/openPMD-api>`_ (`manual <https://openpmd-api.readthedocs.io/>`_)
* data visualization and analysis, including an exploratory Jupyter notebook GUI: `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`_ (`tutorial <https://github.com/openPMD/openPMD-viewer/tree/master/docs/source/tutorials>`__)
* easy visualization of mesh data (fields) using the `plopp <https://scipp.github.io/plopp/>`_ and `scipp <https://github.com/scipp>`_ Python libraries: `openPMD-scipp <https://github.com/pordyna/openpmd_scipp>`_
* straightforward work with particle data as pandas DataFrames: `openPMD pandas support <https://openpmd-api.readthedocs.io/en/latest/analysis/pandas.html>`_
* `yt-project <https://yt-project.org/doc/examining/loading_data.html#openpmd-data>`_
* :ref:`ParaView <pp-paraview>`
* `VisIt <https://github.com/openPMD/openPMD-visit-plugin>`_
* command-line inspection: `openpmd-ls <https://openpmd-api.readthedocs.io/en/latest/utilities/cli.html#openpmd-ls>`_, `bpls <https://adios2.readthedocs.io/en/latest/ecosystem/utilities.html#bpls-inspecting-data>`_ (ADIOS2), and `h5ls <https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html>`_ / `h5dump <https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html>`_ (HDF5)
* `myHDF5 <https://myhdf5.hdfgroup.org/>`_, a web GUI for inspecting HDF5 files based on `H5Web <https://github.com/silx-kit/h5web>`_ (also available as a `VS Code plugin <https://github.com/silx-kit/vscode-h5web>`_)
* converter tools: `openPMD-converter <https://github.com/openPMD/openPMD-converter>`_
* full list of `projects using openPMD <https://github.com/openPMD/openPMD-projects>`_

We recommend trying the openPMD-viewer and openPMD-scipp tools first, followed
by a quick read through the standard itself and the Python part of the
`first read API example <https://openpmd-api.readthedocs.io/en/latest/usage/firstread.html>`_.

If you intend to write your own post-processing routines, make sure to check
out our `example files <https://github.com/openPMD/openPMD-example-datasets>`_
and the `formal, open standard <https://github.com/openPMD/openPMD-standard>`_
on openPMD.

Reading via the openPMD API
---------------------------

When directly using the openPMD API, make yourself familiar with when you need
to call ``series.flush()``.
See `Flush Chunk <https://openpmd-api.readthedocs.io/en/latest/usage/firstread.html#flush-chunk>`_
and `Deferred Data API Contract <https://openpmd-api.readthedocs.io/en/latest/usage/workflow.html#deferred-data-api-contract>`_
for more details.
Forgetting it, or modifying the data before the flush was performed, is a
common user mistake that can lead to very confusing results.

Beyond openPMD
--------------

In addition to the standard openPMD output, PIConGPU offers reduced
diagnostics via :ref:`other plugins <usage-plugins>`, including the very
powerful :ref:`flexible binning plugin <usage-plugins-binningPlugin>`.

.. seealso::
   Interested in writing your own plugin post‑processing tools? See :ref:`development-pytools`.
