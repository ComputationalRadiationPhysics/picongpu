FAQ
===

This file contains frequently asked questions.

   This is more practical compared to :doc:`the misc <misc>`
   documentation section (which discussed fundamental and/or interesting
   concepts). There are no rules for this document, please add much
   stuff.

To learn more about the context of this FAQ please read the
:doc:`translation <translation>` documentation section.

What are PICMI and PyPIConGPU?
------------------------------

**PICMI** is defined `upstream <https://picmi-standard.github.io/>`_
and is a common interface for multiple PIC implementations.

**PyPIConGPU** is a Python module which generates ``.param`` and
``.cfg`` files for PIConGPU.

PICMI is the user interface, and its data structures (objects) are
translated to PyPIConGPU data structures (objects), from which
ultimately ``.param`` and ``.cfg`` files are generated.

As a typical user you will only ever interact with PICMI,
PyPIConGPU is **purely internal**.

PICMI is very lax in its checks, whereas PyPIConGPU is much stricter and
more “trigger-happy” with errors and exceptions.

What does “Testing” mean?
-------------------------

   Please also see :doc:`the respective documentation
   section <testing>`.

To check if a program works by trying some examples on it and comparing
the actual to expected results. Note however, that this does not
*verfiy* the correctness in a formal way.

We try to follow `“Test Driven Development” <https://en.wikipedia.org/wiki/Test-driven_development>`_,
i.e. writing the test before the implementation. This has a couple of
advantages:

-  coverage is kept high
-  “regressions” (a feature stops working) can be caught quickly
-  the programmer is forced to use their code
-  the programmer must think about their problem **both** in terms of
   concrete examples **and** an abstract algorithm
-  the tests serve as an example of how the implementation can be used

What is tested?
---------------

The pipeline from PICMI (user input) to the generated JSON
representation.

If the JSON representation is used correctly is not tested.

   As of the time of writing, this is b/c there is no nice and simple
   framework to check simulation results (in a concise way). If there is
   one when you read this, please add tests checking the compilation
   too.

What are the different types of tests?
--------------------------------------

We differentiate by **quick** and **compiling** tests.

All **quick** tests run in a matter seconds, because they do not compile
anything. The **compiling** tests actually invoke ``pic-build`` and
hence take quite a while to all run.

*Unit* and *Integration* tests are not separated on a structural basis,
both are in the **quick** tests.

How to execute tests?
---------------------

.. code:: bash

   cd $PICSRC/test/python/picongpu
   python -m quick
   python -m compiling

Errors: What types of errors exist and when are they used?
----------------------------------------------------------

Please refer to the `Python
documentation <https://docs.python.org/3/library/exceptions.html>`_ for
available exceptions.

PICMI typically uses ``AssertionError``, in PyPIConGPU ``TypeError`` and
``ValueError`` are the most common.

In which order should one write tests?
--------------------------------------

*Technically speaking* this does not matter, however the recommended
order is:

1. PyPIConGPU
2. PyPIConGPU translation to JSON (by calling ``get_rendering_contex()``, which automatically includes schema check)
3. PICMI to PyPIConGPU translation
4. Compilation (in ``compiling`` tests)

What does it mean to test/use ``get_rendering_context()``?
----------------------------------------------------------

The method ``get_rendering_context()`` is used to retrieve a JSON
representation.

It is generated inside of ``_get_serialized()``, and then the schema is
checked. (See also: :ref:`pypicongpu_faq_schema`)

I.e.: ``get_rendering_context()`` = ``_get_serialized()`` + **schema check**

When you encounter it in tests it is typically used to ensure that the
translation to JSON works.

What does “Variable not initialized” mean, and when does it occur?
------------------------------------------------------------------

When accessing a property that has been created by
``util.build_typesafe_property()``, an error is thrown if this property
is not set yet.

Note that even something as innocent as this can trigger the message:

.. code:: python

   if 0 == len(self.my_list):
       pass

So pay attention to the stack trace and the line numbers.

How to install requirements?
----------------------------

From the repository root execute:

.. code:: bash

   pip install -r requirements.txt

When should a ``picongpu_`` prefix be used for variable names?
--------------------------------------------------------------

Inside of PICMI prefix everything PIConGPU-specific with ``picongpu_``.
In PyPIConGPU should **not** be used.

.. _pypicongpu_faq_schema:

What is a JSON schema?
----------------------

A JSON schema describes how a JSON document may look.

It is used to ensure that PyPIConGPU output is stable: The templates
used for code generation rely on that schema being held.

See `the full spec <https://json-schema.org/>`__ for an in-depth explanation.
