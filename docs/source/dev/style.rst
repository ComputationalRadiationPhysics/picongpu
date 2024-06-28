.. highlight:: cpp

Coding Guidelines
==================

Pre-commit
----------

This project is set up for use with `pre-commit <https://pre-commit.com>`_. Using it will make your code conform with most
of our (easily automatable) code style guidelines automatically. Pre-commit is a tool that manages
`git hooks <https://git-scm.com/docs/githooks>`_ conveniently for you.
In very short (for anything further see `pre-commit <https://pre-commit.com>`_), after running the following in your
working clone of alpaka

.. code-block:: bash

  # if not yet done, install the pre-commit executable following https://pre-commit.com
  cd /path/to/alpaka-working-clone
  pre-commit install

``git`` will run a number of checks prior to every commit and push and will refuse to perform the
pertinent action if they fail. Most of them (like e.g. the formatter) will have automatically altered your working tree
with the necessary changes such that

.. code-block:: bash

  git add -u

will make the next commit pass. Although discouraged, in urgent cases it might be needed to be able to commit even if
the checks fail. For such cases, you can either use

.. code-block:: bash

  git commit --no-verify [...]

to completely skip all checks or use the more fine-grained control described `here <https://pre-commit.com/#temporarily-disabling-hooks>`_.

You can use

.. code-block:: bash

   pre-commit run --all-files

to run all the hooks on all files.

Formatting
----------

Use the ``.clang-format`` file supplied in alpaka's top-level directory to format your code.
This will handle indentation, whitespace and braces automatically.
Checkout ``CONTRIBUTING.md`` on how to run it.

Naming
------

* Types are always in PascalCase (KernelExecCuda, BufT, ...) and singular.
* Variables are always in camelCase (memBufHost, ...) and singular by default. Use plural for collections.
* Namespaces are always in lowercase and singular is preferred.
* Avoid consecutive upper case letters. E.g.: AccOpenMp instead of AccOpenMP, or HtmlRenderer instead of HTMLRenderer.
  This makes names more easily readable.

Type Qualifiers
---------------

The order of type qualifiers should be:
``Type const * const`` for a const pointer to a const Type.
``Type const &`` for a reference to a const Type.

The reason is that types can be read from right to left correctly without jumping back and forth.
``const Type * const`` and ``const Type &`` would require jumping in either way to read them correctly.
clang-format should handle this automatically in most cases.


Variables
---------

* Variables should be initialized at definition to avoid hard to debug errors, even in performance critical code.
  If you suspect a slowdown, measure first.
* Variables should be ``const`` to make the code more easy to understand.
  This is equivalent to functional programming and the SSA (static single assignment) style.
* Prefer direct-initialization using braces for variable definitions, e.g. ``T t{...}``,
  over copy-initialization, e.g. ``T t = {...}``.
  Avoid direct-initialization using parenthesis, e.g. ``T t(...)``.

Comments
--------

* Always use C++-style comments ``//``

Functions
---------

* Always use the trailing return type syntax, even if the return type is ``void``:

.. code-block::

   auto func() -> bool

* This leads to a consistent style for constructs where there is no alternative style (lambdas, functions templates with dependent return types) and standard functions.

Templates
---------

* Template parameters, which are not a single letter, are prefixed with ``T`` to differentiate them from class or function local aliases.

.. code-block:: c++

   template<int I, typename TParam, typename TArgs...>
   auto func() -> bool

* Always use ``typename`` instead of ``class`` for template parameters.
  There is NO semantic difference between them, but ``typename`` matches the intent better.


Traits
------

* Trait classes must have one additional template parameter (defaulted to ``void``) then required to enable SFINAE in specializations:

.. code-block::

   template<typename T, typename TSfinae = void>
   struct GetOffsets;

* Traits for implementations always have the same name as the accessor function but in PascalCase while the member function is camelCase again: ``sin(){...}`` and ``Sin{sin(){...}};``
