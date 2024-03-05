/* Copyright 2022 Jiří Vyskočil, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/IsArrayOrVector.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

TEST_CASE("isArrayOrVector", "[meta]")
{
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<std::array<int, 10>>::value);
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<std::vector<float>>::value);

    [[maybe_unused]] float arrayFloat[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<decltype(arrayFloat)>::value);
}

TEST_CASE("isActuallyNotArrayOrVector", "[meta]")
{
    float notAnArrayFloat = 15.0f;
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<decltype(notAnArrayFloat)>::value);

    [[maybe_unused]] float* notAnArrayFloatPointer = &notAnArrayFloat;
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<decltype(notAnArrayFloatPointer)>::value);

    std::string notAnArrayString{"alpaka"};
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<decltype(notAnArrayString)>::value);
}

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
TEST_CASE("isArrayOrVectorCudaWrappers", "[meta]")
{
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<alpaka::meta::CudaVectorArrayWrapper<double, 1>>::value);
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<alpaka::meta::CudaVectorArrayWrapper<unsigned, 2>>::value);
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<alpaka::meta::CudaVectorArrayWrapper<int, 3>>::value);
    STATIC_REQUIRE(alpaka::meta::IsArrayOrVector<alpaka::meta::CudaVectorArrayWrapper<float, 4>>::value);
}

TEST_CASE("isNotArrayOrVectorCudaVector", "[meta]")
{
    STATIC_REQUIRE_FALSE(alpaka::meta::IsArrayOrVector<uint4>::value);
}
#endif
