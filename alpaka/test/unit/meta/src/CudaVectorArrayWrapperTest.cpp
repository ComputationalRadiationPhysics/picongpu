/* Copyright 2022 Jiří Vyskočil, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */


#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/math/FloatEqualExact.hpp>
#    include <alpaka/meta/CudaVectorArrayWrapper.hpp>
#    include <alpaka/meta/IsStrictBase.hpp>
#    include <alpaka/rand/Traits.hpp>
#    include <alpaka/test/KernelExecutionFixture.hpp>
#    include <alpaka/test/acc/TestAccs.hpp>

#    include <catch2/catch_template_test_macros.hpp>
#    include <catch2/catch_test_macros.hpp>

#    include <type_traits>

/* The tests here use equals for comparing float values for exact equality. This is not
 * an issue of arithmetics. We are testing whether the values saved in a container are the same as the ones retrieved
 * from it afterwards. In this case, returning a value that would not be exactly but only approximately equal to the
 * one that was stored in the container would be a grave error.
 */
template<typename T1, typename T2>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC bool equals(T1 a, T2 b)
{
    return a == static_cast<T1>(b);
}

template<>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC bool equals<float, float>(float a, float b)
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

template<>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC bool equals<double, double>(double a, double b)
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

template<typename T>
class CudaVectorArrayWrapperTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        using T1 = alpaka::meta::CudaVectorArrayWrapper<T, 1>;
        T1 t1{0};
        static_assert(T1::size == 1, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size_v<T1> == 1, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same_v<decltype(t1[0]), T&>, "CudaVectorArrayWrapper in-kernel type test failed!");
        ALPAKA_CHECK(*success, equals(t1[0], T{0}));

        using T2 = alpaka::meta::CudaVectorArrayWrapper<T, 2>;
        T2 t2{0, 1};
        static_assert(T2::size == 2, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size_v<T2> == 2, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same_v<decltype(t2[0]), T&>, "CudaVectorArrayWrapper in-kernel type test failed!");
        ALPAKA_CHECK(*success, equals(t2[0], T{0}));
        ALPAKA_CHECK(*success, equals(t2[1], T{1}));

        using T3 = alpaka::meta::CudaVectorArrayWrapper<T, 3>;
        T3 t3{0, 0, 0};
        t3 = {0, 1, 2};
        static_assert(T3::size == 3, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size_v<T3> == 3, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same_v<decltype(t3[0]), T&>, "CudaVectorArrayWrapper in-kernel type test failed!");
        ALPAKA_CHECK(*success, equals(t3[0], T{0}));
        ALPAKA_CHECK(*success, equals(t3[1], T{1}));
        ALPAKA_CHECK(*success, equals(t3[2], T{2}));

        using T4 = alpaka::meta::CudaVectorArrayWrapper<T, 4>;
        T4 t4{0, 0, 0, 0};
        t4[1] = 1;
        t4[2] = t4[1] + 1;
        t4[3] = t4[2] + t2[1];
        static_assert(T4::size == 4, "CudaVectorArrayWrapper in-kernel size test failed!");
        static_assert(std::tuple_size_v<T4> == 4, "CudaVectorArrayWrapper in-kernel tuple_size test failed!");
        static_assert(std::is_same_v<decltype(t4[0]), T&>, "CudaVectorArrayWrapper in-kernel type test failed!");
        ALPAKA_CHECK(*success, equals(t4[0], T{0}));
        ALPAKA_CHECK(*success, equals(t4[1], T{1}));
        ALPAKA_CHECK(*success, equals(t4[2], T{2}));
        ALPAKA_CHECK(*success, equals(t4[3], T{3}));
    }
};

TEMPLATE_LIST_TEST_CASE("cudaVectorArrayWrapperDevice", "[meta]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    CudaVectorArrayWrapperTestKernel<int> kernelInt;
    REQUIRE(fixture(kernelInt));

    CudaVectorArrayWrapperTestKernel<unsigned> kernelUnsigned;
    REQUIRE(fixture(kernelUnsigned));

    CudaVectorArrayWrapperTestKernel<float> kernelFloat;
    REQUIRE(fixture(kernelFloat));

    CudaVectorArrayWrapperTestKernel<double> kernelDouble;
    REQUIRE(fixture(kernelDouble));
}

TEST_CASE("cudaVectorArrayWrapperHost", "[meta]")
{
    // TODO: It would be nice to check all possible type vs. size combinations.

    using Float1 = alpaka::meta::CudaVectorArrayWrapper<float, 1>;
    Float1 floatWrapper1{-1.0f};
    STATIC_REQUIRE(Float1::size == 1);
    STATIC_REQUIRE(std::tuple_size_v<Float1> == 1);
    STATIC_REQUIRE(std::is_same_v<decltype(floatWrapper1[0]), float&>);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<float1, Float1>::value);
    REQUIRE(equals(floatWrapper1[0], -1.0f));

    using Int1 = alpaka::meta::CudaVectorArrayWrapper<int, 1>;
    Int1 intWrapper1 = {-42};
    STATIC_REQUIRE(Int1::size == 1);
    STATIC_REQUIRE(std::tuple_size_v<Int1> == 1);
    STATIC_REQUIRE(std::is_same_v<decltype(intWrapper1[0]), int&>);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<int1, Int1>::value);
    REQUIRE(intWrapper1[0] == -42);

    using Uint2 = alpaka::meta::CudaVectorArrayWrapper<unsigned, 2>;
    Uint2 uintWrapper2{0u, 1u};
    STATIC_REQUIRE(Uint2::size == 2);
    STATIC_REQUIRE(std::tuple_size_v<Uint2> == 2);
    STATIC_REQUIRE(std::is_same_v<decltype(uintWrapper2[0]), unsigned&>);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<uint2, Uint2>::value);
    REQUIRE(uintWrapper2[0] == 0u);
    REQUIRE(uintWrapper2[1] == 1u);

    using Uint4 = alpaka::meta::CudaVectorArrayWrapper<unsigned, 4>;
    Uint4 uintWrapper4{0u, 0u, 0u, 0u};
    STATIC_REQUIRE(Uint4::size == 4);
    STATIC_REQUIRE(std::tuple_size_v<Uint4> == 4);
    STATIC_REQUIRE(std::is_same_v<decltype(uintWrapper4[0]), unsigned&>);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<uint4, Uint4>::value);
    uintWrapper4[1] = 1u;
    uintWrapper4[2] = uintWrapper4[1] + 1u;
    uintWrapper4[3] = uintWrapper4[2] + uintWrapper2[1];
    REQUIRE(uintWrapper4[0] == 0u);
    REQUIRE(uintWrapper4[1] == 1u);
    REQUIRE(uintWrapper4[2] == 2u);
    REQUIRE(uintWrapper4[3] == 3u);

    using Double3 = alpaka::meta::CudaVectorArrayWrapper<double, 3>;
    Double3 doubleWrapper3{0.0, 0.0, 0.0};
    doubleWrapper3 = {0.0, -1.0, -2.0};
    STATIC_REQUIRE(Double3::size == 3);
    STATIC_REQUIRE(std::tuple_size_v<Double3> == 3);
    STATIC_REQUIRE(std::is_same_v<decltype(doubleWrapper3[0]), double&>);
    STATIC_REQUIRE(alpaka::meta::IsStrictBase<double3, Double3>::value);
    REQUIRE(equals(doubleWrapper3[0], 0.0));
    REQUIRE(equals(doubleWrapper3[1], -1.0));
    REQUIRE(equals(doubleWrapper3[2], -2.0));
}

#endif
