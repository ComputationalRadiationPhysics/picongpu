Sphinx
======

.. sectionauthor:: Axel Huebl, alpaka-group

In the following section we explain how to contribute to this documentation.

If you are reading the `HTML version <https://alpaka.readthedocs.io>`_ and want to improve or correct existing pages, check the "*Edit on GitHub*" link on the right upper corner of each document.

Alternatively, go to `docs/source` in our source code and follow the directory structure of `reStructuredText`_ (``.rst``) files there.
For intrusive changes, like structural changes to chapters, please open an issue to discuss them beforehand.

.. _reStructuredText: https://www.sphinx-doc.org/en/stable/rest.html

Build Locally
-------------

This document is build based on free open-source software, namely `Sphinx`_, `Doxygen`_ (C++ APIs as XML), `Breathe`_ (to include doxygen XML in Sphinx) and `rst2pdf`_ (render the cheat sheet).
A web-version is hosted on `ReadTheDocs`_.

.. _Sphinx: https://github.com/sphinx-doc/sphinx
.. _Doxygen: http://doxygen.org
.. _Breathe: https://github.com/michaeljones/breathe
.. _rst2pdf: https://rst2pdf.org/
.. _ReadTheDocs: https://readthedocs.org/

The following requirements need to be installed (once) to build our documentation successfully:

.. code-block:: bash

    cd docs/

    # doxygen is not shipped via pip, install it externally,
    # from the homepage, your package manager, conda, etc.
    # example:
    sudo apt-get install doxygen
    # sudo pacman -S doxygen

    # python tools & style theme
    pip install -r requirements.txt # --user


With all documentation-related software successfully installed, just run the following commands to build your docs locally.
Please check your documentation build is successful and renders as you expected before opening a pull request!

.. code-block:: bash

    # skip this if you are still in docs/
    cd docs/

    # parse the C++ API documentation (default: xml format)
    doxygen Doxyfile

    # render the cheatsheet.pdf
    rst2pdf -s cheatsheet/cheatsheet.style source/basic/cheatsheet.rst -o cheatsheet/cheatsheet.pdf

    # render the '.rst' files with sphinx
    make html

    # open it, e.g. with firefox :)
    firefox build/html/index.html

    # now again for the pdf :)
    make latexpdf

    # open it, e.g. with okular
    build/latex/alpaka.pdf

.. hint::

   Run `make clean` to clean the build directory before executing actual make. This is necessary to reflect changes outside the rst files.

.. hint::

   There is a checklinks target to check links in the rst files on availability:

   .. code-block:: bash

      # check existence of links
      # cd docs/
      make checklinks

.. hint::

   The Doxyfile for doxygen is configured to output in xml format per default.
   Another targets can be configured in the Doxyfile. The final documentations are stored in ``docs/doxygen/``.

   .. code-block:: bash

      # run in docs/doxygen/
      sed -i -E 's/(GENERATE_HTML\s*=\s*)NO/\1YES/g' Doxyfile

readthedocs
-----------

To maintain or import a github project an account on `ReadTheDocs`_ is required.
Further instructions can be found on `readthedocs on github <https://github.com/readthedocs/readthedocs.org>`_ and `readthedocs import guide <https://docs.readthedocs.io/en/stable/intro/import-guide.html>`_.

Useful Links
------------

 * `A primer on writing reStructuredText files for sphinx <https://www.sphinx-doc.org/en/stable/rest.html>`_
 * `Why You Shouldn't Use "Markdown" for Documentation <https://www.ericholscher.com/blog/2016/mar/15/dont-use-markdown-for-technical-docs/>`_
 * `reStructuredText vs. Markdown <https://eli.thegreenplace.net/2017/restructuredtext-vs-markdown-for-technical-documentation/>`_
 * `Markdown Limitations in Sphinx <https://docs.readthedocs.io/en/latest/intro/getting-started-with-sphinx.html#using-markdown-with-sphinx>`_
