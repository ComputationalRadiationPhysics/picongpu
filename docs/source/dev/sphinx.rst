.. _development-sphinx:

Sphinx
======

.. sectionauthor:: Axel Huebl, Marco Garten

In the following section we explain how to contribute to this documentation.

If you are reading the HTML version on http://picongpu.readthedocs.io and want to improve or correct existing pages, check the "Edit on GitHub" link on the right upper corner of each document.

Alternatively, go to `docs/source` in our source code and follow the directory structure of `reStructuredText`_ (``.rst``) files there.
For intrusive changes, like structural changes to chapters, please open an issue to discuss them beforehand.

.. _reStructuredText: http://www.sphinx-doc.org/en/stable/rest.html

Build Locally
-------------

This document is build based on free open-source software, namely `Sphinx`_, `Doxygen`_ (C++ APIs as XML) and `Breathe`_ (to include doxygen XML in Sphinx).
A web-version is hosted on `ReadTheDocs`_.

.. _Sphinx: https://github.com/sphinx-doc/sphinx
.. _Doxygen: http://doxygen.org
.. _Breathe: https://github.com/michaeljones/breathe
.. _ReadTheDocs: https://readthedocs.org/

The following requirements need to be installed (once) to build our documentation successfully:

.. code-block:: bash

    cd docs/

    # doxygen is not shipped via pip, install it externally,
    # from the homepage, your package manager, conda, etc.
    # example:
    sudo apt-get install doxygen

    # python tools & style theme
    pip install -r requirements.txt # --user

In order to not break any of your existing Python configurations, you can also create a new environment that you only use for building the documentation.
Since it is possible to install doxygen with conda, the following demonstrates this.

.. code-block:: bash

    cd docs/

    # create a bare conda environment containing just all the requirements
    # for building the picongpu documentation
    # note: also installs doxygen inside this environment
    conda env create --file picongpu-docs-env.yml

    # start up the environment as suggested during its creation e.g.
    conda activate picongpu-docs-env
    # or
    source activate picongpu-docs-env

With all documentation-related software successfully installed, just run the following commands to build your docs locally.
Please check your documentation build is successful and renders as you expected before opening a pull request!

.. code-block:: bash

    # skip this if you are still in docs/
    cd docs/

    # parse the C++ API documentation,
    #   enjoy the doxygen warnings!
    doxygen
    # render the `.rst` files and replace their macros within
    #   enjoy the breathe errors on things it does not understand from doxygen :)
    make html

    # open it, e.g. with firefox :)
    firefox build/html/index.html

    # now again for the pdf :)
    make latexpdf

    # open it, e.g. with okular
    build/latex/PIConGPU.pdf


Useful Links
------------

 * `A primer on writing restFUL files for sphinx <http://www.sphinx-doc.org/en/stable/rest.html>`_
 * `Why You Shouldn't Use "Markdown" for Documentation <http://ericholscher.com/blog/2016/mar/15/dont-use-markdown-for-technical-docs/>`_
 * `Markdown Limitations in Sphinx <https://docs.readthedocs.io/en/latest/getting_started.html#in-markdown>`_
