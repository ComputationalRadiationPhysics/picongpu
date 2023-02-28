Testing Strategy
================

This document aims to explain which components are tested to which
degree, and more generally how pypicongpu tests should be treated.

The extent of the tests tries to balance multiple factors:

-  more tests means more work (and more maintenance down the road)
-  fewer test mean more errors go unnoticed
-  more important/fragile components should be tested more rigorously
   (“important” here means “could create a lot trouble down the road if
   it does not work as specified”)
-  some type of checks require a lot of code (which needs to be
   maintained), others do not

How these factors are traded against each other is layed out for each
module below.

All of the tests are built to get broken – run them frequently and
expand them if you add functionality! Some of the tests are rather
rigid, so they might fail even if they are not supposed to. When
encountering a failing test act with care and adjust the test case if
that is plausible.

.. note::

   For an explanation on the python ``unittests`` module please refer to the `python manual <https://docs.python.org/3.8/library/unittest.html>`_.

   For more notes on tools support (including test coverage) see :ref:`pypicongpu-misc-toolsupport`.

Structure
---------

All tests are stored in the directory ``test/python/picongpu/``. For convenience the
tests are separated into several categories (stored in the respective
directories):

-  ``quick``: tests which run in a short amount of time (a couple of
   seconds, typically unit tests)
-  ``compiling``: tests which run a longer amount of time (mainly the runner)
-  ``e2e``: tests which tests picongpu end-to-end

Inside of each of these directories the structure of the python source
(``lib/python/picongpu/``) is replicated.

To execute the tests run their ``__main__.py`` by invoking from
``test/python/picongpu/`` (they don't work from anywhere else!):
repository root:

.. code:: bash

   python -m quick
   python -m compiling
   python -m e2e

Append ``--help`` for options. E.g. for quick debugging use
``python -m quick -f --locals``. Individual test cases can be
accessed with their class name, and optional method name:

.. code:: bash

   python -m quick TestElement
   python -m quick TestElement.test_periodic_table_names

.. note:

    All compile test runs create setup directories at /tmp/<userName>. Please make sure that
    they are removed afterwards, thank you.

.. note:

    The path to each generated compile test picongpu output is printed to console.
    If you encounter a failing compiling test, go to that directory and run a pic-build by
    hand to get a standard picongpu compile output for debugging.

.. note::

   The tests are loaded by using ``from SOMEWHERE import *`` -- which is bad style.
   For that reason all ``__init__.py`` files in the tests have the style checks disabled with ``# flake8: noqa``.
   When changing the test runner (maybe in the future) these skipped checks should be abandoned.

PICMI
-----

Any passed PICMI object is assumed to be correct. PICMI (upstream) does
not necessarily enforce this – if that should be enforced please add
checks upstream with the picmistandard instead of adding them in
pypicongpu.

The method ``get_as_pypicongpu()`` is only tested to a shallow level:
Most test cases are (significantly) longer than the assertions/checks,
yet would usually only check if the values are copied correctly.

Such tests would achieve only a small benefit yet be fairly long and are
hence kept to a minimum.

PyPIConGPU objects
------------------

The most important functionality of pypicongpu objects is storing data
in a defined type and ensuring that mandatory parameters are provided.
Both are ensured by using a utility function to generate
getters/setters, and hence there is a pretty low chance to make a
mistake there.

Yet there are some sanity checks performed, including the translation to
CPP objects. As these translations are tightly coupled to the used
``.param`` file (templates), they could change fairly frequently and are
as such not subject to rigerous testing.

PIConGPU project template
-------------------------

The ``.param`` and ``.cfg`` files where the params defined by pypicongpu
are placed inside. These are not tested separately, as such tests would
have to be pretty much equivalent to the end-to-end tests.

PIConGPU (and its build process)
--------------------------------

PIConGPU and its build process (using ``pic-build``) are assumed to be
correct when used as documented.

End-To-End
----------

   TODO: This functionality is not yet implemented

To verify the PICMI input standard there are a few examples provided for
which both a PICMI input and an equivalent set of ``.param`` files
exists.

Their respective outputs are compared programmatically, tiny differences
are ignored, anything else is considered a failure.

As maintaining both ``.param`` and PICMI input files takes a lot of time
and effort, there only exist a couple of such examples. To compensate
for their low number, these examples themselves are fairly complicated.
