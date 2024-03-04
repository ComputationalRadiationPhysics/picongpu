.. highlight:: cpp

Coding Guidelines
==================

.. attention::
   The Coding Guidelines are currently revised

General
-------

* Use the ``.clang-format`` file supplied in alpaka's top-level directory to format your code. This will handle indentation,
whitespace and braces automatically. Usage:

.. code-block:: bash

  clang-format-16 -i <sourcefile>

* If you want to format the entire code base execute the following command from alpaka's top-level directory:

.. code-block:: bash

  find example include test -name '*.hpp' -o -name '*.cpp' | xargs clang-format-16 -i

Windows users should use `Visual Studio's native clang-format integration
<https://devblogs.microsoft.com/cppblog/clangformat-support-in-visual-studio-2017-15-7-preview-1/>`.

Naming
------

* Types are always in PascalCase (KernelExecCuda, BufT, ...) and singular.
* Variables are always in camelCase (memBufHost, ...) and plural for collections and singular else.
* Namespaces are always in lowercase and singular is preferred.
* There are no two consecutive upper case letters (AccOpenMp, HtmlRenderer, IoHandler, ...). This makes names more easily readable.


Types
-----

* Always use integral types with known width (``int32_t``, ``uint64_t``, ...).
  Never use ``int``, ``unsigned long``, etc.


Type Qualifiers
---------------

The order of  type qualifiers should be:
``Type const * const`` for a const pointer to a const Type.
``Type const &`` for a reference to a const Type.

The reason is that types can be read from right to left correctly without jumping back and forth.
``const Type * const`` and ``const Type &`` would require jumping in either way to read them correctly.


Variables
---------

* Variables should always be initialized on construction because this can produce hard to debug errors.
  This can (nearly) always be done even in performance critical code without sacrificing speed by using a functional programming style.
* Variables should (nearly) always be ``const`` to make the code more easy to understand.
  This is equivalent to functional programming and the SSA (static single assignment) style used by LLVM.
  This should have no speed implication as every half baked compiler analyses the usage of variables and reuses registers.
* Variable definitions should be differentiated from assignments by using either ``(...)`` or ``{...}`` but never ``=`` for definitions.
  Use ``uint32_t const iUsageOfThisVariable(42);`` instead of ``uint32_t const iUsageOfThisVariable = 42;``


Comments
--------

* Always use C++-Style comments ``//``
* For types use
  ``//#############################################################################``
  to start the comment block.
* For functions use
  ``//-----------------------------------------------------------------------------``
  to start the comment block.


Functions
---------

* Always use the trailing return type syntax with the return type on a new line even if the return type is void:

.. code-block::

   auto func()
   -> bool

* This makes it easier to see the return type because it is on its own line.
* This leads to a consistent style for constructs where there is no alternative style (lambdas, functions templates with dependent return types) and standard functions.
* Each function parameter is on a new indented line:

.. code-block::

   auto func(
       float f1,
       float f2)
   -> bool
   {
       return true
   }

.. code-block::

   func(
       1.0f,
       2.0f);

* Makes it easier to see how many parameters there are and which position they have.


Templates
---------

* Template parameters are prefixed with ``T`` to differentiate them from class or function local typedefs.
* Each template parameter is on a new indented line:

.. code-block:: c++

   template<
       typename TParam,
       typename TArgs...>
   auto func()
   -> bool

* Makes it easier to see how many template parameters there are and which position they have.
* Always use ``typename`` for template parameters. There is NO difference to class and typename matches the intent better.


Traits
------

* Trait classes always have one more template parameter (with default parameter) then is required for enabling SFINAE in the specialization:

.. code-block::

   template<
       typename T,
       typename TSfinae = void>
   struct GetOffsets;

* Template trait aliases always end with a ``T`` e.g. ``BufT`` while the corresponding trait ends with ``Type`` e.g. ``BufType``
* Traits for implementations always have the same name as the accessor function but in PascalCase while the member function is camelCase again: ``sin(){...}`` and ``Sin{sin(){...}};``

Includes
--------

* The order of includes is from the most specialized header to the most general one.
  This order helps to find missing includes in more specialized headers because the general ones are always included afterwards.
* A comment with the types or functions included by a include file make it easier to find out why a special header is included.
