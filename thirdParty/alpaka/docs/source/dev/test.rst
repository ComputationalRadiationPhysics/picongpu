.. highlight:: cpp

Writing a unit test
===================

After implementing a new functionality in Alpaka, it is recommended to test it. Indeed, having no compile time errors
when building alpaka with any backend doesn't mean that such functionality is well implemented and behaves as expected.

Unit tests are written and integrated with `Catch2 <https://github.com/catchorg/Catch2>`_ and they are standalone executables located in the ``test/unit`` and
``test/integ`` paths inside the ``build`` folder. They can be built all at once by setting the ``CMake``
argument ``-DBUILD_TESTING=ON`` and running the command ``cmake --build .`` in the ``build`` folder, and then run with the
``ctest`` command as described in the :doc:`Installation section </basic/install>`.

In alternative, a target test can be built by adding the flag ``--target <testname>`` to the ``cmake --build`` command and
then executed as a normal binary file (``./<testname>``) in the corresponding folder mentioned above.

Sample workflow
---------------

As a sample workflow, consider the ``getWarpSize`` functionality that, given an ``alpaka device``, returns the number
of threads that constitute a warp.

First of all, the unit test should be located inside a folder under the ``test/unit`` path (namely ``dev``). Such a folder
will contain a ``CMakeLists.txt`` file and a ``src`` folder, in which the ``.cpp`` file with the test itself will be located.
Furthermore, it is necessary to add the folder in which the test is located to ``test/unit/CMakeLists.txt``:

.. code-block::
    add_subdirectory("dev/")

to make sure that ``CMake`` will build the test written in the ``.cpp`` file under that directory.

After these steps, it all comes to writing the *actual* test. First of all, the necessary header libraries should be
included the ``.cpp`` file:

.. code-block:: c++
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/test/acc/TestAccs.hpp>
    #include <catch2/catch.hpp>

The first ``include`` statement targets specifically the header in which the functionality is included; with the second one
only accelerators corresponding to enabled backends are selected for testing, while the third one is necessary to integrate
Catch2.

Test cases using an alpaka accelerator are then typically introduced with the ``TEMPLATE_LIST_TEST_CASE`` macro (as described
`here <https://github.com/catchorg/Catch2/blob/devel/docs/test-cases-and-sections.md#type-parametrised-test-cases>`_).
It takes three arguments:
* a free form test name (must be unique)
* a tag
* the accelerator(s) that must run the test case(s) (i.e. ``alpaka::test::TestAccs`` targets all the accelereators selected
by the ``TestAccs`` header).

Some aliases might be useful:

.. code-block:: c++
    using Dev = alpaka::Dev<TestType>;
    using Platform = alpaka::Platform<Dev>;

The template parameter for ``alpaka::Dev`` is ``TestType``, which basically target the specific accelerator that corresponds
to the backend which is executing the test.

Since the function that must be tested has a device object as argument, it is necessary to define one:

.. code-block:: c++
    auto const platform = Platform{};
    Dev const dev = alpaka::getDevByIdx(platform, 0);

Having all the necessary objects well defined, the function can be called and the result stored in an object:

.. code-block:: c++
    auto const warpExtent = alpaka::getWarpSize(dev);

Finally, the condition under which the test succeeds should be defined with a Catch2 macro.
The most used macros are: ``REQUIRE`` and ``CHECK`` family.
From the `Catch2 documentation <https://github.com/catchorg/Catch2/blob/devel/docs/assertions.md>`_:

*The REQUIRE family of macros tests an expression and aborts the test case if it fails. The CHECK family are
equivalent but execution continues in the same test case even if the assertion fails.*

For example:

.. code-block:: c++
    REQUIRE(warpExtent > 0);

evaluates the expression between the round brackets and if an exception is thrown, it is caught, reported, and counted
as a failure. Such expression depends on what must be actually tested (i.e. the size of the warp can never be equal or
less than 0).
Furthermore, additional, compile-time expressions can be evaluated as well:

.. code-block:: c++
    STATIC_REQUIRE(std::is_same_v<decltype(warpExtent), int>);

which checks that the type of the warp size is integer (cannot be float). Differently from the ``REQUIRE`` macro,
if the expression is false, the ``STATIC_REQUIRE`` will throw an error at compile time, without running the application. 
