.. _usage-workflows-numberOfCells:

Setting the Number of Cells
---------------------------

.. sectionauthor:: Axel Huebl

Together with the grid resolution in :ref:`grid.param <usage-params-core>`, the number of cells in our :ref:`.cfg files <usage-tbg>` determine the overall size of a simulation (box).
The following rules need to be applied when setting the number of cells:

Each device needs to:

#. contain an integer *multiple* of supercells
#. at least *three* supercells
#. for non periodic boundary conditions, the number of absorbing boundary cells for devices at the simulation boundary (see :ref:`grid.param <usage-params-core>`) must fit into the local volume

The grid size will be automatically adjusted if the conditions above are not fulfilled. 
This behavior can be disabled by using the command line option ``--autoAdjustGrid off``

Supercell sizes in terms of number of cells are set in :ref:`memory.param <usage-params-memory>` and are by default ``8x8x4`` for 3D3V simulations on GPUs.
For 2D3V simulations, ``16x16`` is usually a good supercell size, however the default is simply cropped to ``8x8``, so make sure to change it to get more performance.
