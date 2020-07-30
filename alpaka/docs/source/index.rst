:orphan:

.. only:: html

  .. image:: ../logo/alpaka.svg

.. only:: latex

  .. image:: ../logo/alpaka.pdf

*alpaka - An Abstraction Library for Parallel Kernel Acceleration*

The alpaka library is a header-only C++14 abstraction library for accelerator development. Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

.. CAUTION::
   The readthedocs pages are work in progress and contain outdated sections.

alpaka - How to Read This Document
----------------------------------

Generally, **follow the manual pages in-order** to get started.
Individual chapters are based on the information of the chapters before.

.. only:: html

   The online version of this document is **versioned** and shows by default the manual of the last *stable* version of alpaka.
   If you are looking for the latest *development* version, `click here <https://alpaka.readthedocs.io/en/latest/>`_.

.. note::

   Are you looking for our latest Doxygen docs for the API?

   - See https://alpaka-group.github.io/alpaka/


.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1

   install/instructions
   install/HIP

.. toctree::
   :caption: USAGE
   :maxdepth: 1

   usage/intro
   usage/abstraction
   usage/implementation
   usage/cmake_example
   usage/cheatsheet

.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1

   dev/style
   dev/sphinx
   API Reference <https://alpaka-group.github.io/alpaka>

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

