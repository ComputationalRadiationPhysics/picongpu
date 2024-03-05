/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/test/Array.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSharedMemStNonNullTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(6, 0, 0)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Waddress" // warning: the compiler can assume that the address of 'a' will never be NULL [-Waddress]
#endif
        // Multiple runs to make sure it really works.
        for(std::size_t i = 0u; i < 10; ++i)
        {
            auto& a = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &a);

            auto& b = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &b);

            auto& c = alpaka::declareSharedVar<float, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<float*>(nullptr) != &c);

            auto& d = alpaka::declareSharedVar<double, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<double*>(nullptr) != &d);

            auto& e = alpaka::declareSharedVar<std::uint64_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint64_t*>(nullptr) != &e);


            auto& f = alpaka::declareSharedVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &f[0]);

            auto& g = alpaka::declareSharedVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != &g[0]);

            auto& h = alpaka::declareSharedVar<alpaka::test::Array<double, 16>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, static_cast<double*>(nullptr) != &h[0]);
        }
#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(6, 0, 0)
#    pragma GCC diagnostic pop
#endif
    }
};

TEMPLATE_LIST_TEST_CASE("nonNull", "[blockSharedMemSt]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Use multiple threads to make sure the synchronization really works.
    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(3u)));

    BlockSharedMemStNonNullTestKernel kernel;

    REQUIRE(fixture(kernel));
}

class BlockSharedMemStSameTypeAdressVerificationTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Allocations in the loop with same template signature should be equal too this allocation.
        // 21 byte is chosen to break nice alignment for all following calls of declareSharedVar.
        auto& baseAllocation = alpaka::declareSharedVar<alpaka::test::Array<std::uint8_t, 21>, 42>(acc);

        // Multiple runs to make sure it really works.
        for(std::size_t i = 0u; i < 10; ++i)
        {
            // same type but different id should result in different pointer for each allocation
            auto& a = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            auto& b = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &a != &b);
            auto& c = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &b != &c);
            ALPAKA_CHECK(*success, &a != &c);
            ALPAKA_CHECK(*success, &b != &c);

            auto& d = alpaka::declareSharedVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &a != &d[0]);
            ALPAKA_CHECK(*success, &b != &d[0]);
            ALPAKA_CHECK(*success, &c != &d[0]);
            auto& e = alpaka::declareSharedVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
            ALPAKA_CHECK(*success, &a != &e[0]);
            ALPAKA_CHECK(*success, &b != &e[0]);
            ALPAKA_CHECK(*success, &c != &e[0]);
            ALPAKA_CHECK(*success, &d[0] != &e[0]);
        }

        for(std::size_t i = 0u; i < 10; ++i)
        {
            // same type and same id should result in same pointer for each allocation
            auto& a = alpaka::declareSharedVar<alpaka::test::Array<std::uint8_t, 21>, 42>(acc);
            auto& b = alpaka::declareSharedVar<alpaka::test::Array<std::uint8_t, 21>, 42>(acc);
            ALPAKA_CHECK(*success, &a == &b);
            ALPAKA_CHECK(*success, &a == &baseAllocation);

            auto& lastAllocation = alpaka::declareSharedVar<alpaka::test::Array<std::uint8_t, 23>, 23>(acc);
            auto& c = alpaka::declareSharedVar<alpaka::test::Array<std::uint8_t, 23>, 23>(acc);
            ALPAKA_CHECK(*success, &lastAllocation == &c);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("sameTypeAddressVerification", "[blockSharedMemSt]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Use multiple threads to make sure the synchronization really works.
    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(3u)));

    BlockSharedMemStSameTypeAdressVerificationTestKernel kernel;

    REQUIRE(fixture(kernel));
}
