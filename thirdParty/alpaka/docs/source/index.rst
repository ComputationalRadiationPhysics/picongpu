:orphan:

.. only:: html

  .. image:: ../logo/alpaka.svg

.. only:: latex

  .. image:: ../logo/alpaka.png

*alpaka - An Abstraction Library for Parallel Kernel Acceleration*

The alpaka library is a header-only C++17 abstraction library for accelerator development.
Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

.. CAUTION::
   The readthedocs pages are provided with best effort, but may contain outdated sections.

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
   :caption: Basic
   :maxdepth: 1

   basic/intro.rst
   basic/install.rst
   basic/example.rst
   basic/abstraction.rst
   basic/library.rst
   basic/cheatsheet.rst

.. toctree::
   :caption: Advanced
   :maxdepth: 1

   advanced/rationale.rst
   advanced/mapping.rst
   advanced/cmake.rst
   advanced/compiler.rst

.. toctree::
   :caption: Extra Info
   :maxdepth: 1

   info/similar_projects.rst

.. toctree::
   :caption: Development
   :maxdepth: 1

   dev/backends.rst
   dev/details.rst
   dev/style
   dev/test.srt
   dev/sphinx
   dev/ci
   API Reference <https://alpaka-group.github.io/alpaka>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
