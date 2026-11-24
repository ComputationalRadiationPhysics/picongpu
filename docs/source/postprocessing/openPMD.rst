.. _pp-openPMD:

openPMD
=======

.. sectionauthor:: Axel Huebl
.. moduleauthor:: Axel Huebl

.. tip::

	This section collects links to external resources for the openPMD data standard.
	For a description of PIConGPU's openPMD plugin, refer to :ref:`the plugin documentation <usage-plugins-openPMD>`, or to the :ref:`legacy interface tutorial <hemeraIn5min>`.


If you hear about *openPMD** for the first time you can find a quick `online tutorial <http://www.openpmd.org>`_ on it here.

As a user of PIConGPU, you will be mainly interested in our :ref:`python tools <usage-python-utils>` and readers, that can read openPMD, e.g. into:

* read & write data: `openPMD-api <https://github.com/openPMD/openPMD-api>`_ (`manual <https://openpmd-api.readthedocs.io/>`_)
* visualization and analysis, including an exploratory Jupyter notebook GUI: `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`_ (`tutorial <https://github.com/openPMD/openPMD-viewer/tree/master/tutorials>`__)
* `yt-project <http://yt-project.org/doc/examining/loading_data.html#openpmd-data>`_ (`tutorial <https://gist.github.com/C0nsultant/5808d5f61b271b8f969d5c09f5ca91dc>`__)
* :ref:`ParaView <pp-paraview>`
* `VisIt <https://github.com/openPMD/openPMD-visit-plugin>`_
* converter tools: `openPMD-converter <https://github.com/openPMD/openPMD-converter>`_
* full list of `projects using openPMD <https://github.com/openPMD/openPMD-projects>`_

If you intend to write your own post-processing routines, make sure to check out our `example files <https://github.com/openPMD/openPMD-example-datasets>`_, the `formal, open standard <https://github.com/openPMD/openPMD-standard>`_ on openPMD and a `list of projects <https://github.com/openPMD/openPMD-projects>`_ that already support openPMD.

.. seealso::
   Interested in writing your own plugin post‑processing tools? See :ref:`development-pytools`.
