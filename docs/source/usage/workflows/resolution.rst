.. _usage-workflows-resolution:

Changing the Resolution with a Fixed Target
-------------------------------------------

.. sectionauthor:: Axel Huebl

One often wants to refine an already existing resolution in order to model a setup more precisely or to be able to model a higher density.

#. change cell sizes and time step in :ref:`simulation.param <usage-params-core>`
#. change number of GPUs in :ref:`.cfg file <usage-tbg>`
#. change number of :ref:`number of cells and distribution over GPUs <usage-workflows-numberOfCells>` in :ref:`.cfg file <usage-tbg>`
#. adjust (transveral) positioning of targets in :ref:`density.param <usage-params-core>`
#. :ref:`recompile <usage-basics>`
