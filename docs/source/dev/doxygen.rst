.. _development-doxygen:

Doxygen
=======

.. sectionauthor:: Axel Huebl

Although our main documentation is integrated via :ref:`Sphinx <development-sphinx>`, some developers might still want to read the plain HTML build of Doxygen.
If you need these docs, this section explains how to build it locally.

Requirements
------------

First, install `Doxygen`_ and its dependencies for graph generation.

.. _Doxygen: http://doxygen.org

.. code-block:: bash

    # install requirements (Debian/Ubuntu)
    sudo apt-get install doxygen graphviz

    # enable HTML output in our Doxyfile
    sed -i 's/GENERATE_HTML.*=.*NO/GENERATE_HTML     = YES/' docs/Doxyfile

Build
-----

Now run the following commands to build the Doxygen HTML documentation locally.

.. code-block:: bash

    cd docs/

    # build the doxygen HTML documentation
    doxygen

    # open the generated HTML pages, e.g. with firefox
    firefox html/index.html
