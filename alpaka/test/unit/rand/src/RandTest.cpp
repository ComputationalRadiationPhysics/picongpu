/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/rand/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

class RandTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            auto dist = alpaka::rand::distribution::createNormalReal<float>(acc);
            [[maybe_unused]] auto const r = dist(gen);
            if constexpr(!BOOST_ARCH_PTX)
                ALPAKA_CHECK(*success, std::isfinite(r));
        }

        {
            auto dist = alpaka::rand::distribution::createNormalReal<double>(acc);
            [[maybe_unused]] auto const r = dist(gen);
            if constexpr(!BOOST_ARCH_PTX)
                ALPAKA_CHECK(*success, std::isfinite(r));
        }
        {
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, 0.0f <= r);
            ALPAKA_CHECK(*success, 1.0f > r);
        }

        {
            auto dist = alpaka::rand::distribution::createUniformReal<double>(acc);
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, 0.0 <= r);
            ALPAKA_CHECK(*success, 1.0 > r);
        }

        {
            auto dist = alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc);
            [[maybe_unused]] auto const r = dist(gen);
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // default generator for accelerator
        auto genDefault = alpaka::rand::engine::createDefault(acc, 12345u, 6789u);
        genNumbers(acc, success, genDefault);

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
        // TODO: These ifdefs are wrong: They will reduce the test to the
        // smallest common denominator from all enabled backends
        // std::random_device
        auto genRandomDevice = alpaka::rand::engine::createDefault(alpaka::rand::RandomDevice{}, 12345u, 6789u);
        genNumbers(acc, success, genRandomDevice);

        // MersenneTwister
        auto genMersenneTwister = alpaka::rand::engine::createDefault(alpaka::rand::MersenneTwister{}, 12345u, 6789u);
        genNumbers(acc, success, genMersenneTwister);

        // TinyMersenneTwister
        auto genTinyMersenneTwister
            = alpaka::rand::engine::createDefault(alpaka::rand::TinyMersenneTwister{}, 12345u, 6789u);
        genNumbers(acc, success, genTinyMersenneTwister);
#endif
    }
};

TEMPLATE_LIST_TEST_CASE("defaultRandomGeneratorIsWorking", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    RandTestKernel kernel;

    REQUIRE(fixture(kernel));
}

//! Helper trait to check if the given accelerator is HIP
template<typename TAcc>
struct IsAccHIP : public std::false_type
{
};

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template<typename TDim, typename TIdx>
struct IsAccHIP<alpaka::AccGpuHipRt<TDim, TIdx>> : public std::true_type
{
};
#endif

TEMPLATE_LIST_TEST_CASE("defaultRandomGeneratorIsTriviallyCopyable", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using DefaultEngine = decltype(alpaka::rand::engine::createDefault(std::declval<Acc>(), 0u, 0u, 0u));
    constexpr auto isEngineTriviallyCopyable = std::is_trivially_copyable_v<DefaultEngine>;
    // For older HIP versions the internal HIPrand/ROCrand state was not trivially copyable.
    // This causes alpaka rand state for the HIP accelerator and those versions to also not be trivially copyable.
    // It was fixed on AMD side in https://github.com/ROCmSoftwarePlatform/rocRAND/pull/252.
    // Thus we guard the test to skip HIP accelerator and older HIP versions.
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && (BOOST_LANG_HIP < BOOST_VERSION_NUMBER(5, 2, 0))
    if constexpr(!IsAccHIP<Acc>::value)
        STATIC_REQUIRE(isEngineTriviallyCopyable);
#else
    STATIC_REQUIRE(isEngineTriviallyCopyable);
#endif
}
