Misc
====

This is a catch-all section for general notes on PyPIConGPU.

Python Concepts
---------------

Some common concepts recurr throughout the code and are not explained in
detail when used. This aims to give a non-exhaustive list of such
patterns.

   They are as close to “standard python” as possible, so using a search
   engine/asking a python expert should also help you understand them.

Type Checking with Typeguard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use as defined in `PEP 484 “Type Hints” <https://peps.python.org/pep-0484/>`_.
Briefly:

.. code:: python

   def greeting(name: str) -> str:
       return 'Hello ' + name

Note that these type annotations are **not checked by python** on their own.

Typechecks are enabled through
`typeguard <https://typeguard.readthedocs.io/>`__, mostly using the
annotation ``@typeguard.typechecked`` for classes.

**This does not check attribute type annotations, see next section.**

Typesafe Class Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^

Class attributes can use type annotations according to PEP 484, but this
is not enforced by typeguard.

The solution is to use `properties <https://docs.python.org/3.10/library/functions.html#property>`_.

In the code you will see:

.. code:: python

   class SomeObject:
       attr = util.build_typesafe_property(int)

Now the following hold:

1. Accessing ``attr`` before it has been set will **raise an exception**
2. Assigning ``attr`` anything other than an ``int`` will raise a ``TypeError``

..

   Note: instead of types you can use specifications provided by
   ``import typing``, e.g. ``typing.Optional[int]``. Optional vars still
   have **no default**, so you have to set them to ``None`` explicitly
   if you don’t want to provide them.

Internally the declaration above is expanded to the equivalent of:

.. code:: python

   @typeguard.typechecked
   def getter(self) -> int:
       if not hasattr(self, "actual_name_for_var__attr"):
           raise AttributeError("variable is not initialized")
       return getattr(self, "actual_name_for_var__attr")
       
   @typeguard.typechecked
   def setter(self, value: int) -> None:
       setattr(self, "actual_name_for_var__attr", value)

   class SomeObject:
       attr = property(getter, setter)

Map-Filter-Reduce
^^^^^^^^^^^^^^^^^

``map()``, ``filter()``, and ``reduce()`` are “higher-order functions”
and a basis of the functional programming paradigm (as opposed to loops
for iterative processing). You can find an an introduction
`here <https://www.learnpython.org/en/Map,_Filter,_Reduce>`_ (but any
other intro is probably fine too).

Boils down to:

.. code:: python

   [2, 4, 6] == list(map(lambda x: 2*x,
                         [1, 2, 3]))

..

   ``map()`` does not return a list directly and must be cast again.
   (Python, why?)

``map()`` does not work on dictionaries, to for this the dictionary is
cast to a set of key-value tuples:

.. code:: python

   {"a": 3, "b": 0} == dict(map(
       lambda kv_pair: (kv_pair[0], len(kv_pair[1])),
       {"a": [0, 0, 0], "b": []}.items()))

..

   Note: Do not feel obliged to follow this pattern.
   It is commonly used, because it allows a concise notation,
   yet it is very much **not** mandatory.


.. _pypicongpu-misc-toolsupport:

Tool Support
------------

This is a short list of tools to aid you with development.

Formatting
^^^^^^^^^^

`PEP 8 <https://peps.python.org/pep-0008/>`_ (formatting guidelines) compliance:

run from repo root ``flake8 lib/python/ test/python/``

Note: Please obey the line length from PEP 8 even if that is annoying at times,
this makes viewing files side-by-side much simpler.

Run Tests
^^^^^^^^^

go into ``test/python/picongpu``,
execute ``python -m quick``
(for tests with compilation ``python -m compiling``)

Test Coverage
^^^^^^^^^^^^^

0. install `coverage.py <https://coverage.readthedocs.io/>`_: ``pip install coverage``
1. go to tests: ``cd test/python/picongpu``
2. execute tests, record coverage: ``coverage run --branch -m quick``
3. view reports

   - for PyPIConGPU: ``find ../../../lib/python/picongpu/pypicongpu/ -name '*.py' | xargs coverage report -m``
   - for PICMI: ``find ../../../lib/python/picongpu/picmi/ -name '*.py' | xargs coverage report -m``
   - Goal is 100% coverage (missing sections are printed)

For further options see `the coverage.py doc <https://coverage.readthedocs.io/>`_.

.. _pypicongpu-misc-apidoc:

API Documentation
-----------------

Document the API using docstrings,
which will be rendered by `Sphinx AutoAPI <https://sphinx-autoapi.readthedocs.io/>`_ automatically.
The result is available from the TOC, although the name is generated to be the python module name: :doc:`autoapi/picongpu/pypicongpu/index`.

The configuration is placed in Sphinx's ``conf.py``,
all classes are traversed by using ``__all__`` defined in **every** ``__init__.py``.

Additional checks (completeness etc.) are **NOT** employed (and not even possible).
So pay attention that you document everything you need.

As everything is passed to Sphinx, 
**docstring should be valid** `reStructuredText <https://docutils.sourceforge.io/rst.html>`_.
(`reStructuredText quick guide from Sphinx <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_)

You may use `Sphinx-exclusive directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html>`_ too,
e.g. ``:ref:`MARK```.

To document parameters, return values etc. make them compatible with Sphinx's autodoc.
For details please refer to `the respective documentation section <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_.

.. note::

   If there is an reStructuredText syntax error during generation (printed as warning after sphinx invocation ``make html``) to debug:

   1. visit the respective page
   2. click *View page source* at the top of the page

      (This might be only availble **locally**, the online version displays *Edit on GitHub* instead)

An example can be found in `the official documentation <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_.
The types are implictly given by the `type hints <https://www.python.org/dev/peps/pep-0484/>`_,
so these can be omitted:

.. literalinclude:: ./doc_example.py
   :language: python

Note that ``:raises TypeError:`` is omitted too,
because all functions are checked for types anyways.

This produces the following code section:

.. autofunction:: doc_example.my_func
