.. _development-doxygen:

Doxygen
=======

.. sectionauthor:: Axel Huebl

An online version of our Doxygen build can be found at

http://computationalradiationphysics.github.io/picongpu

We regularly update it via

.. code-block:: bash

   git checkout gh-pages

   # optional argument: branch or tag name
   ./update.sh

   git commit -a
   git push

This section explains what is done when this script is run to build it manually.

Requirements
------------

install `Doxygen`_ and its dependencies for graph generation.

.. _Doxygen: http://doxygen.org

.. code-block:: bash

    # install requirements (Debian/Ubuntu)
    sudo apt-get install doxygen graphviz

Activate HTML Output
--------------------

activate the generation of html files in the doxygen config file

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
