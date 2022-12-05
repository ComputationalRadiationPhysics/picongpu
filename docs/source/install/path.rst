.. _install-path:

Introduction
============

.. sectionauthor:: Axel Huebl

Installing PIConGPU means :ref:`installing C++ libraries <install-dependencies>` that PIConGPU depends on and :ref:`setting environment variables <install-profile>` to find those dependencies.
The first part is usually the job of a system administrator while the second part needs to be configured on the user side.

Depending on your experience, role, computing environment and expectations for optimal hardware utilization, you have several ways to install and select PIConGPU's dependencies.
Choose your favorite *install and environment management method* below, young padawan, and follow the corresponding sections of the next chapters.

Ways to Install
---------------

Choose *one* of the installation methods below to get started.

Load Modules
^^^^^^^^^^^^

On HPC systems and clusters, software is usually provided by system administrators via a module system (e.g. [modules]_, [Lmod]_).
In case our :ref:`software dependencies <install-dependencies>` are available, we usually create a file in our ``$HOME`` named :ref:`<queueName>_picongpu.profile <install-profile>`.
It loads according modules and sets :ref:`helper environment variables <install-dependencies-picongpu>`.

.. important::

   For many HPC systems we have already prepared and maintain an environment which will run out of the box.
   See if your system is :ref:`in the list <install-profile>` so you can skip the installation completely!

Build from Source
^^^^^^^^^^^^^^^^^

You choose a supported C++ compiler and configure, compile and install all missing dependencies from source.
You are responsible to manage the right versions and configurations.
Performance will be ideal if architecture is chosen correctly (and/or if built directly on your hardware).
You then set environment variables to find those installs.

Conda
^^^^^

We currently do not have an official conda install (yet).
Due to pre-build binaries, performance could be not ideal and HPC cluster support (e.g. MPI) might be very limited.
Useful for small desktop or single-node runs.

References
----------

.. [modules]
        J.L. Furlani, P.W. Osel.
        *Abstract Yourself With Modules*,
        `Proceedings of the 10th USENIX conference on System administration (1996) <http://modules.sourceforge.net/docs/absmod.pdf>`_,
        http://modules.sourceforge.net

.. [Lmod]
        R. McLay and contributors.
        *Lmod: An Environment Module System based on Lua, Reads TCL Modules, Supports a Software Hierarchy*,
        https://github.com/TACC/Lmod
