.. _install-source:

.. seealso::

   You will need to understand how to use `the terminal <http://www.ks.uiuc.edu/Training/Tutorials/Reference/unixprimer.html>`_.

.. note::

   This section is a short introduction in case you are missing a few software packages, want to try out a cutting edge development version of a software or have no system administrator or software package manager to build and install software for you.

From Source
-----------

.. sectionauthor:: Axel Huebl

Don't be afraid, young physicist, self-compiling C/C++ projects is easy, fun and profitable!

Building a project from source essentially requires three steps:

    #. configure the project and find its dependencies
    #. compile the project
    #. install the project

All of the above steps can be performed without administrative rights ("root" or "superuser") as long as the install is not targeted at a system directory (such as ``/usr``) but inside a user-writable directory (such as ``$HOME`` or a project directory).

Preparation
^^^^^^^^^^^

In order to compile projects from source, we assume you have individual directories created to store *source code*, *build temporary files* and *install* the projects to:

.. code-block:: bash

    # source code
    mkdir $HOME/src
    # temporary build directory
    mkdir $HOME/build
    # install target for dependencies
    mkdir $HOME/lib

Note that on some supercomputing systems, you might need to install the final software outside of your home to make dependencies available during run-time (when the simulation runs).
Use a different path for the last directory then.

What is Compiling?
^^^^^^^^^^^^^^^^^^

.. note::

   This section is **not** yet the installation of PIConGPU from source.
   It just introduces in general how one compiles projects.

   If you like to skip this introduction, :ref:`jump straight to the dependency install section <install-dependencies>`.

Compling can differ in two principle ways: building *inside* the source directory ("in-source") and in a *temporary directory* ("out-of-source").
Modern projects prefer the latter and use a build system such as [CMake]_.

An example could look like this

.. code-block:: bash

   # go to an empty, temporary build directory
   cd $HOME/build
   rm -rf ../build/*
   
   # configurate, build and install into $HOME/lib/project
   cmake -DCMAKE_INSTALL_PREFIX=$HOME/lib/project $HOME/src/project_to_compile
   make
   make install

Often, you want to pass further options to CMake with ``-DOPTION=VALUE`` or modify them interactively with ``ccmake .`` after running the initial cmake command.
The second step which compiles the project can in many cases be parallelized by ``make -j``.
In the final install step, you might need to prefix it with ``sudo`` in case ``CMAKE_INSTALL_PREFIX`` is pointing to a system directory.

Some older projects often build *in-source* and use a build system called *autotools*.
The syntax is still very similar:

.. code-block:: bash

   # go to the source directory of the project
   cd $HOME/src/project_to_compile
   
   # configurate, build and install into $HOME/lib/project
   configure --prefix=$HOME/lib/project
   make
   make install

One can usually pass further options with ``--with-something=VALUE`` or ``--enable-thing`` to ``configure``.
See ``configure --help`` when installing an *autotools* project.

That is all on the theory of building projects from source!

Now Start
^^^^^^^^^

You now know all the basics to install from source.
Continue with the following section to :ref:`build our dependencies <install-dependencies>`.

References
^^^^^^^^^^

.. [CMake]
        Kitware Inc.
        *CMake: Cross-platform build management tool*,
        https://cmake.org/
