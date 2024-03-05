/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

// Base kernel, not to be used directly in unit tests
struct KernelWithOmpScheduleBase
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        // No run-time check is performed
        ALPAKA_CHECK(*success, true);
    }
};

// Kernel that sets the schedule kind via constexpr ompScheduleKind, but no chunk size.
// Checks that this variable is only declared and not defined, It also tests that alpaka never odr-uses it.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithConstexprMemberOmpScheduleKind : KernelWithOmpScheduleBase
{
    static constexpr auto ompScheduleKind = TKind;
};

// Kernel that sets the schedule kind via non-constexpr ompScheduleKind, but no chunk size.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithMemberOmpScheduleKind : KernelWithOmpScheduleBase
{
    static const alpaka::omp::Schedule::Kind ompScheduleKind = TKind;
};

// Kernel that sets the schedule chunk size via constexpr ompScheduleChunkSize in addition to schedule kind, but no
// chunk size. Checks that this variable is only declared and not defined, It also tests that alpaka never odr-uses it.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithConstexprStaticMemberOmpScheduleChunkSize : KernelWithConstexprMemberOmpScheduleKind<TKind>
{
    static constexpr int ompScheduleChunkSize = 5;
};

// Kernel that sets the schedule chunk size via non-constexpr ompScheduleChunkSize in addition to schedule kind.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithStaticMemberOmpScheduleChunkSize : KernelWithMemberOmpScheduleKind<TKind>
{
    static int const ompScheduleChunkSize;
};

// In this case, the member has to be defined externally
template<alpaka::omp::Schedule::Kind TKind>
int const KernelWithStaticMemberOmpScheduleChunkSize<TKind>::ompScheduleChunkSize = 2;

// Kernel that sets the schedule chunk size via non-constexpr non-static ompScheduleChunkSize in addition to schedule
// kind.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithMemberOmpScheduleChunkSize : KernelWithConstexprMemberOmpScheduleKind<TKind>
{
    int ompScheduleChunkSize = 4;
};

// Kernel that copies the given base kernel and adds an OmpSchedule trait on top
template<typename TBase>
struct KernelWithTrait : TBase
{
};

namespace alpaka::trait
{
    // Specialize the trait for kernels of type KernelWithTrait<>
    template<typename TBase, typename TAcc>
    struct OmpSchedule<KernelWithTrait<TBase>, TAcc>
    {
        template<typename TDim, typename... TArgs>
        ALPAKA_FN_HOST static auto getOmpSchedule(
            KernelWithTrait<TBase> const& /* kernelFnObj */,
            Vec<TDim, Idx<TAcc>> const& /* blockThreadExtent */,
            Vec<TDim, Idx<TAcc>> const& /* threadElemExtent */,
            TArgs const&... /* args */) -> alpaka::omp::Schedule
        {
            return alpaka::omp::Schedule{alpaka::omp::Schedule::Static, 4};
        }
    };
} // namespace alpaka::trait

// Generic testing routine for the given kernel type
template<typename TAcc, typename TKernel>
void testKernel()
{
    using Acc = TAcc;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    // Base version with no OmpSchedule trait
    TKernel kernel;
    REQUIRE(fixture(kernel));

    // Same members, but with OmpSchedule trait
    KernelWithTrait<TKernel> kernelWithTrait;
    REQUIRE(fixture(kernelWithTrait));
}

// Note: it turned out not possible to test all possible combinations as it causes several compilers to crash in CI.
// However the following tests should cover all important cases

TEMPLATE_LIST_TEST_CASE("kernelWithOmpScheduleBase", "[kernel]", alpaka::test::TestAccs)
{
    testKernel<TestType, KernelWithOmpScheduleBase>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstexprMemberOmpScheduleKind", "[kernel]", alpaka::test::TestAccs)
{
    testKernel<TestType, KernelWithConstexprMemberOmpScheduleKind<alpaka::omp::Schedule::NoSchedule>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpScheduleKind", "[kernel]", alpaka::test::TestAccs)
{
    testKernel<TestType, KernelWithMemberOmpScheduleKind<alpaka::omp::Schedule::Static>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstexprStaticMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    testKernel<TestType, KernelWithConstexprStaticMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Dynamic>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithStaticMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    testKernel<TestType, KernelWithStaticMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Guided>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
#if defined _OPENMP && _OPENMP >= 200805
    testKernel<TestType, KernelWithMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Auto>>();
#endif
    testKernel<TestType, KernelWithMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Runtime>>();
}
