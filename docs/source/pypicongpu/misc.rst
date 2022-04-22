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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use as defined in `PEP 484 “Type Hints” <https://peps.python.org/pep-0484/>`_.
Briefly:

.. code:: python

   def greeting(name: str) -> str:
       return 'Hello ' + name

Note that these type annotations are **not checked by python** on their own.

Typechecks are enabled through
`typeguard <https://typeguard.readthedocs.io/>`__, mostly using the
annotation ``@typechecked`` for classes.

**This does not check attribute type annotations, see next section.**

Typesafe Class Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~

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

   @typechecked
   def getter(self) -> int:
       if not hasattr(self, "actual_name_for_var__attr"):
           raise AttributeError("variable is not initialized")
       return getattr(self, "actual_name_for_var__attr")
       
   @typechecked
   def setter(self, value: int) -> None:
       setattr(self, "actual_name_for_var__attr", value)

   class SomeObject:
       attr = property(getter, setter)

Map-Filter-Reduce
~~~~~~~~~~~~~~~~~

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
