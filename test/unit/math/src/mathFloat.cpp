/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephans
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

// This file only has unit tests for real numbers in order to split the tests between object files
using FunctorsReal = alpaka::meta::
    Concatenate<mathtest::UnaryFunctorsReal, mathtest::BinaryFunctorsReal, mathtest::TernaryFunctorsReal>;
using TestAccFunctorTuplesReal = alpaka::meta::CartesianProduct<std::tuple, TestAccs, FunctorsReal>;

TEMPLATE_LIST_TEST_CASE("mathOpsFloat", "[math] [operator]", TestAccFunctorTuplesReal)
{
    /*
     * All alpaka::math:: functions are tested here except sincos.
     * The function will be called with a buffer from the custom Buffer class.
     * This argument Buffer contains ArgsItems from Defines.hpp and can be
     * accessed with the overloaded operator().
     * The args Buffer looks similar like [[0, 1], [2, 3], [4, 5]],
     * where every sub-list makes one functor-call so the result Buffer would be:
     * [f(0, 1), f(2, 3), f(4, 5)].
     * The results are saved in a different Buffer witch contains plain data.
     * The results are than compared to the result of a std:: implementation.
     * The default result is nan and should fail a test.
     *
     * BE AWARE that:
     * - ALPAKA_CUDA_FAST_MATH should be disabled
     * - not all casts between float and double can be detected.
     * - no explicit edge cases are tested, rather than 0, maximum and minimum
     *   - but it is easy to add a new Range:: enum-type with custom edge cases
     *  - some tests may fail if ALPAKA_CUDA_FAST_MATH is turned on
     * - nan typically fails every test, but could be normal defined behaviour
     * - inf/-inf typically dont fail a test
     * - for easy debugging the << operator is overloaded for Buffer objects
     * - arguments are generated between 0 and 1000
     *     and the default argument-buffer-extent is 1000
     * The arguments are generated in DataGen.hpp and can easily be modified.
     * The arguments depend on the Range:: enum-type specified for each functor.
     * ----------------------------------------------------------------------
     * - each functor has an arity and a array of ranges
     *     - there is one args Buffer and one results Buffer
     *         - each buffer encapsulated the host/device communication
     *         - as well as the data access and the initialisation
     * - all operators are tested independent, one per kernel
     * - tests the results against the std implementation ( catch REQUIRES)
     *
     * TestKernel
     * - uses the alpaka::math:: option from the functor
     * - uses the device-buffer  option from the args
     *
     * EXTENSIBILITY:
     * - Add new operators in Functor.hpp and add them to the ...Functors tuple.
     * - Add a new Range:: enum-type in Defines.hpp
     *     - specify a fill-method in DataGen.hpp
     * - Add a new Arity:: enum-type in Defines.hpp
     *     - add a matching operator() function in Functor.hpp,
     *     - add a new ...Functors tuple
     *     - call alpaka::meta::forEachType with the tuple in ForEachFunctor
     */

    using Acc = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto testTemplate = mathtest::TestTemplate<Acc, Functor>{};
    testTemplate.template operator()<float>();
}
