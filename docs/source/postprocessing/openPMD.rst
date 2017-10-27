.. _pp-openPMD:

openPMD
=======

.. sectionauthor:: Axel Huebl
.. moduleauthor:: Axel Huebl

Our :ref:`HDF5 <usage-plugins-HDF5>` and :ref:`ADIOS <usage-plugins-ADIOS>` use a specific internal markup to structure physical quantities called **openPMD**.
If you hear of it for the first time you can find a quick `online tutorial <http://www.openpmd.org>`_ on it here.

As a user of PIConGPU, you will be mainly interested in our :ref:`python tools <pp-python>` and readers, that can read openPMD, e.g. into:

* Numpy reader and Jupyter notebook GUI via `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`_ (`tutorial <https://github.com/openPMD/openPMD-viewer/tree/master/tutorials>`_)
* `yt-project <http://yt-project.org/doc/examining/loading_data.html#openpmd-data>`_ (`tutorial <https://gist.github.com/C0nsultant/5808d5f61b271b8f969d5c09f5ca91dc>`_)
* :ref:`ParaView <pp-paraview>`
* `VisIt <https://github.com/openPMD/openPMD-visit-plugin>`_
* full list of `projects using openPMD <https://github.com/openPMD/openPMD-projects>`_

If you intend to write your own post-processing routines, make sure to check out our `example files <https://github.com/openPMD/openPMD-example-datasets>`_, the `formal, open standard <https://github.com/openPMD/openPMD-standard>`_ on openPMD and a `list of projects <https://github.com/openPMD/openPMD-projects>`_ that already support openPMD.
