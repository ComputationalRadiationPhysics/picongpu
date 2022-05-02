/* Copyright 2022 Jiří Vyskočil, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

class FloatEqualExactTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        // Store the comparison result in a separate variable so that the function call is outside ALPAKA_CHECK.
        // In case ALPAKA_CHECK were ever somehow modified to silence the warning by itself.
        bool testValue = false;

        float floatValue = -1.0f;
        testValue = alpaka::math::floatEqualExactNoWarning(floatValue, -1.0f);
        ALPAKA_CHECK(*success, testValue);

        double doubleValue = -1.0;
        testValue = alpaka::math::floatEqualExactNoWarning(doubleValue, -1.0);
        ALPAKA_CHECK(*success, testValue);
    }
};

TEMPLATE_LIST_TEST_CASE("floatEqualExactTest", "[math]", alpaka::test::TestAccs)
{
    // Host tests

    // Store the comparison result in a separate variable so that the function call is outside REQUIRE.
    // In case REQUIRE were ever somehow modified to silence the warning by itself.
    bool testValue = false;

    float floatValue = -1.0;
    testValue = alpaka::math::floatEqualExactNoWarning(floatValue, -1.0f);
    REQUIRE(testValue);

    double doubleValue = -1.0;
    testValue = alpaka::math::floatEqualExactNoWarning(doubleValue, -1.0);
    REQUIRE(testValue);

    // Device tests

    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    FloatEqualExactTestKernel kernelFloat;
    REQUIRE(fixture(kernelFloat));
}
