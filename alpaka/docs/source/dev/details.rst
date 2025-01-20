.. highlight:: cpp

Details
=======

.. image:: /images/structure.png
   :alt: Overview of the structure of the *alpaka* library with concepts and implementations.

The full stack of concepts defined by the *alpaka* library and their inheritance hierarchy is shown in the third column of the preceding figure.
Default implementations for those concepts can be seen in the blueish columns.
The various accelerator implementations, shown in the lower half of the figure, only differ in some of their underlying concepts but can share most of the base implementations.
The default implementations can, but do not have to be used at all.
They can be replaced by user code in arbitrary granularity.
By substituting, for instance, the atomic operation implementation of an accelerator, the execution can be fine-tuned, to better utilize the hardware instruction set of a specific processor.
However, also complete accelerators, devices and all of the other concepts can be implemented by the user without the need to change any part of the *alpaka* library itself.
The way this and other things are implemented is explained in the following paragraphs.

Concept Implementations
-----------------------

The *alpaka* library has been implemented with extensibility in mind.
This means that there are no predefined classes, modeling the concepts, the *alpaka* functions require as input parameters.
They allow arbitrary types as parameters, as long as they model the required concept.

C++ provides a language inherent object oriented abstraction allowing to check that parameters to a function comply with the concept they are required to model.
By defining interface classes, which model the *alpaka* concepts, the user would be able to inherit his extension classes from the interfaces he wants to model and implement the abstract virtual methods the interfaces define.
The *alpaka* functions in turn would use the corresponding interface types as their parameter types.
For example, the ``Buffer`` concept requires methods for getting the pitch or accessing the underlying memory.
With this intrusive object oriented design pattern the ``BufCpu`` or ``BufCudaRt`` classes would have to inherit from an ``IBuffer`` interface and implement the abstract methods it declares.
An example of this basic pattern is shown in the following source snippet:

.. code-block::

   struct IBuffer
   {
     virtual std::size_t getPitch() const = 0;
     virtual std::byte * data() = 0;
     ...
   };

   struct BufCpu : public IBuffer
   {
     virtual std::size_t getPitch() const override { ... }
     virtual std::byte * data() override { ... }
     ...
   };

   ALPAKA_FN_HOST auto copy(
     IBuffer & dst,
     IBuffer const & src)
   -> void
   {
     ...
   }

The compiler can then check at compile time that the objects the user wants to use as function parameters can be implicitly cast to the interface type, which is the case for inherited base classes.
The compiler returns an error message on a type mismatch.
However, if the *alpaka* library were using those language inherent object oriented abstractions, the extensibility and optimizability it promises would not be possible.
Classes and run-time polymorphism require the implementer of extensions to intrusively inherit from predefined interfaces and override special virtual functions.

This is feasible for user defined classes or types where the source code is available and where it can be changed.
The ``std::vector`` class template on the other hand would not be able to model the ``Buffer`` concept because we can not change its definition to inherit from the ``IBuffer`` interface class since it is part of the standard library.
The standard inheritance based object orientation of C++ only works well when all the code it is to interoperate with can be changed to implement the interfaces.
It does not enable interaction with unalterable or existing code that is too complex to change, which is the reality in the majority of software projects.

Another option to implement an extensible library is to follow the way the C++ standard library uses.
It allows to specialize function templates for user types to model concepts without altering the types themselves.
For example, the ``std::begin`` and ``std::end`` free function templates can be specialized for user defined types.
With those functions specialized, the C++11 range-based for loops (``for(auto & i : userContainer){...}``) see *C++ Standard 6.5.4/1* can be used with user defined types.
Equally specializations of ``std::swap`` and other standard library function templates can be defined to extend those with support for user types.
One Problem with function specialization is, that only full specializations are allowed.
A partial function template specialization is not allowed by the standard.
Another problem can emerge due to users carelessly overloading the template functions instead of specializing them.
Mixing function overloading and function template specialization on the same base template function can result in unexpected results.
The reasons and effects of this are described more closely in an article from H. Sutter (currently convener of the ISO C++ committee) called *Sutter's Mill: Why Not Specialize Function Templates?* in the *C/C++ Users Journal* in July 2001.

.. seealso::
   `different way <http://ericniebler.com/2014/10/21/customization-point-design-in-c11-and-beyond/>`_

The solution given in the article is to provide *"a single function template that should never be specialized or overloaded"*.
This function simply forwards its arguments *"to a class template containing a static function with the same signature"*.
This template class can fully or partially be specialized without affecting overload resolution.

The way the *alpaka* library implements this is by not using the C++ inherent object orientation but lifting those abstractions to a higher level.
Instead of using a non-extensible``class``/``struct`` and abstract virtual member functions for the interface, *alpaka* defines free functions.
All those functions are templates allowing the user to call them with arbitrary self defined types and not only those inheriting from a special interface type.
Unlike member functions, they have no implicit ``this`` pointer, so the object instance has to be explicitly given as a parameter.
Overriding the abstract virtual interface methods is replaced by the specialization of a template type that is defined for each such function.

A concept is completely implemented by specializing the predefined template types.
This allows to extend and fine-tune the implementation non-intrusively.
For example, the corresponding pitch and memory pinning template types can be specialized for ``std::vector``.
After doing this, the ``std::vector`` can be used everywhere a buffer is accepted as argument throughout the whole *alpaka* library without ever touching its definition.

A simple function allowing arbitrary tasks to be enqueued into a queue can be implemented in the way shown in the following code.
The ``TSfinae`` template parameter will be explained in a `following section <#Template-Specialization-Selection-on-Arbitrary-Conditions>`_.

.. code-block::

   namespace alpaka
   {
     template<
       typename TQueue,
       typename TTask,
       typename TSfinae = void>
     struct Enqueue;

     template<
       typename TQueue,
       typename TTask>
     ALPAKA_FN_HOST auto enqueue(
       TQueue & queue,
       TTask & task)
     -> void
     {
       Enqueue<
         TQueue,
         TTask>
       ::enqueue(
         queue,
         task);
     }
   }

A user who wants his queue type to be used with this ``enqueue`` function has to specialize the ``Enqueue`` template struct.
This can be either done partially by only replacing the ``TQueue`` template parameter and accepting arbitrary tasks or by fully specializing and replacing both ``TQueue`` and ``TTask``. This gives the user complete freedom of choice.
The example given in the following code shows this by specializing the ``Enqueue`` type for a user queue type ``UserQueue`` and arbitrary tasks.

.. code-block::

   struct UserQueue{};

   namespace alpaka
   {
     // partial specialization
     template<
       typename TTask>
     struct Enqueue<
       UserQueue
       TTask>
     {
       ALPAKA_FN_HOST static auto enqueue(
         UserQueue & queue,
         TTask & task)
       -> void
       {
         //...
       }
     };
   }

In addition the subsequent code shows a full specialization of the ``Enqueue`` type for a given ``UserQueue`` and a ``UserTask``.

.. code-block::

   struct UserQueue{};
   struct UserTask{};

   namespace alpaka
   {
     // full specialization
     template<>
     struct Enqueue<
       UserQueue
       UserTask>
     {
       ALPAKA_FN_HOST static auto enqueue(
         UserQueue & queue,
         UserTask & task)
       -> void
       {
         //...
       }
     };
   }

When the ``enqueue`` function template is called with an instance of ``UserQueue``, the most specialized version of the ``Enqueue`` template is selected depending on the type of the task ``TTask`` it is called with.

A type can model the queue concept completely by defining specializations for ``alpaka::Enqueue`` and ``alpaka::Empty``.
This functionality can be accessed by the corresponding ``alpaka::enqueue`` and ``alpaka::empty`` template functions.

Currently there is no native language support for describing and checking concepts in C++ at compile time.
A study group (SG8) is working on the ISO `specification for concepts <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4377.pdf>`_ and compiler forks implementing them do exist.
For usage in current C++ there are libraries like `Boost.ConceptCheck <https://www.boost.org/doc/libs/1_58_0/libs/concept_check/concept_check.htm>`_ which try to emulate requirement checking of concept types.
Those libraries often exploit the preprocessor and require non-trivial changes to the function declaration syntax.
Therefore the *alpaka* library does not currently make use of *Boost.ConceptCheck*.
Neither does it facilitate the proposed concept specification due to its dependency on non-standard compilers.

The usage of concepts as described in the working draft would often dramatically enhance the compiler error messages in case of violation of concept requirements.
Currently the error messages are pointing deeply inside the stack of library template invocations where the missing method or the like is called.
Instead of this, with concept checking it would directly fail at the point of invocation of the outermost template function with an expressive error message about the parameter and its violation of the concept requirements.
This would simplify especially the work with extendable template libraries like *Boost* or *alpaka*.
However, in the way concept checking would be used in the *alpaka* library, omitting it does not change the semantic of the program, only the compile time error diagnostics.
In the future when the standard incorporates concept checking and the major compilers support it, it will be added to the *alpaka* library.


Template Specialization Selection on Arbitrary Conditions
---------------------------------------------------------

Basic template specialization only allows for a selection of the most specialized version where all explicitly stated types have to be matched identically.
It is not possible to enable or disable a specialization based on arbitrary compile time expressions depending on the parameter types.
To allow such conditions, *alpaka* adds a defaulted and unused ``TSfinae`` template parameter to all declarations of the implementation template structs.
This was shown using the example of the ``Enqueue`` template type.
The C++ technique called SFINAE, an acronym for *Substitution failure is not an error* allows to disable arbitrary specializations depending on compile time conditions.
Specializations where the substitution of the parameter types by the deduced types would result in invalid code will not result in a compile error, but will simply be omitted.
An example in the context of the ``Enqueue`` template type is shown in the following code.

.. code-block::

   struct UserQueue{};

   namespace alpaka
   {
     template<
       typename TQueue,
       typename TTask>
     struct Enqueue<
       TQueue
       TTask,
       std::enable_if_t<
         std::is_base_of_v_<UserQueue, TQueue>
         && (TTask::TaskId == 1u)
       >>
     {
       ALPAKA_FN_HOST static auto enqueue(
         TQueue & queue,
         TTask & task)
       -> void
       {
         //...
       }
     };
   }

The ``Enqueue`` specialization shown here does not require any direct type match for the ``TQueue`` or the ``TTask`` template parameter.
It will be used in all contexts where ``TQueue`` has inherited from ``UserQueue`` and where the ``TTask`` has a static const integral member value ``TaskId`` that equals one.
If the ``TTask`` type does not have a ``TaskId`` member, this code would be invalid and the substitution would fail.
However, due to SFINAE, this would not result in a compiler error but rather only in omitting this specialization.
The ``std::enable_if`` template results in a valid expression, if the condition it contains evaluates to true, and an invalid expression if it is false.
Therefore it can be used to disable specializations depending on arbitrary boolean conditions.
It is utilized in the case where the ``TaskId`` member is unequal one or the ``TQueue`` does not inherit from ``UserQueue``.
In this circumstances, the condition itself results in valid code but because it evaluates to false, the ``std::enable_if`` specialization results in invalid code and the whole ``Enqueue`` template specialization gets omitted.

Argument dependent lookup for math functions
--------------------------------------------

Alpaka comes with a set of basic mathematical functions in the namespace `alpaka::math`.
These functions are dispatched in two ways to support user defined overloads of these functions.

Let's take `alpaka::math::abs` as an example:
When `alpaka::math::abs(acc, value)` is called, a concrete implementation of `abs` is picked via template specialization.
Concretely, something similar to `alpaka::math::trait::Abs<decltype(acc), decltype(value)>{}(acc, value)` is called.
This allows alpaka (and the user) to specialize the template `alpaka::math::trait::Abs` for various backends and various argument types.
E.g. alpaka contains specializations for `float` and `double`.
If there is no specialization within alpaka (or by the user), the default implementation of `alpaka::math::trait::Abs<....>{}(acc, value)` will just call `abs(value)`.
This is called an unqualified call and C++ will try to find a function called `abs` in the namespace where the type of `value` is defined.
This feature is called Argument Dependent Lookup (ADL).
Using ADL for types which are not covered by specializations in alpaka allows a user to bring their own implementation for which `abs` is meaningful, e.g. a custom implementation of complex numbers or a fixed precision type.
