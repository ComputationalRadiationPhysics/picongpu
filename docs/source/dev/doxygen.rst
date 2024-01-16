.. _development-doxygen:

Doxygen
=======

.. sectionauthor:: Axel Huebl

PIConGPU uses `Doxygen` for API documentation. Please provide the corresponding annotations in new and updated code as needed. To build this documentation do the following:

Requirements
------------

Install `Doxygen`_ and its dependencies for graph generation.

.. _Doxygen: http://doxygen.org

.. code-block:: bash

    # install requirements (Debian/Ubuntu)
    sudo apt-get install doxygen graphviz

Activate HTML Output
--------------------

Activate the generation of html files in the doxygen config file

.. code-block:: bash

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
