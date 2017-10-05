.. _install-spack:

.. seealso::

   You will need to understand how to use `the terminal <http://www.ks.uiuc.edu/Training/Tutorials/Reference/unixprimer.html>`_.

.. warning::

   Docker images are experimental and not yet fully automated or integrated.

Docker
------

.. sectionauthor:: Axel Huebl

Preparation
^^^^^^^^^^^

First `install nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ for your distribution.

Install
^^^^^^^

The download of a pre-configured image with the latest version of PIConGPU is now as easy as:

.. code-block:: bash

   nvidia-docker pull ax3l/picongpu

Use PIConGPU
^^^^^^^^^^^^

Start a pre-configured LWFA live-simulation with

.. code-block:: bash

   nvidia-docker run -p 2459:2459 -t ax3l/picongpu:0.3.0 /bin/bash -lc start_lwfa
   # open firefox and isaac client

or just open the container and run your own:

.. code-block:: bash

   nvidia-docker run -it ax3l/picongpu

.. note::

   PIConGPU can also run *without a GPU*!
   We will provide more image variants in the future.
