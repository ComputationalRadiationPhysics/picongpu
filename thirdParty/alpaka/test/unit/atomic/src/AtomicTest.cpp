/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, Sergei Bastrakov, René Widera, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <type_traits>

template<typename T1, typename T2>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto equals(T1 a, T2 b) -> bool
{
    return a == b;
}

ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto equals(float a, float b) -> bool
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto equals(double a, double b) -> bool
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicAdd(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = static_cast<T>(operandOrig + value);
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicAdd(acc, &operand, value, alpaka::hierarchy::Threads{});
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicSub(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = static_cast<T>(operandOrig - value);
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicSub>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicSub(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicMin(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = (operandOrig < value) ? operandOrig : value;
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicMin>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicMin(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicMax(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = (operandOrig > value) ? operandOrig : value;
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicMax>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicMax(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicExch(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = value;
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicExch>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicExch(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicInc(TAcc const& acc, bool* success, T operandOrig) -> void
{
    // \TODO: Check reset to 0 at 'value'.
    T const value = static_cast<T>(42);
    T const reference = static_cast<T>(operandOrig + 1);
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicInc>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicInc(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicDec(TAcc const& acc, bool* success, T operandOrig) -> void
{
    // \TODO: Check reset to 'value' at 0.
    T const value = static_cast<T>(42);
    T const reference = static_cast<T>(operandOrig - 1);
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicDec>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicDec(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicAnd(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = operandOrig & value;
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicAnd>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicAnd(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicOr(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    T const reference = operandOrig | value;
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicOr>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOr(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicXor(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(operandOrig + static_cast<T>(4));
    T const reference = operandOrig ^ value;
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOp<alpaka::AtomicXor>(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicXor(acc, &operand, value);
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicCas(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);

    // with match
    {
        T const compare = operandOrig;
        T const reference = value;
        {
            operand = operandOrig;
            T const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operandOrig;
            T const ret = alpaka::atomicCas(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
    }

    // without match
    {
        T const compare = static_cast<T>(operandOrig + static_cast<T>(1));
        T const reference = operandOrig;
        {
            operand = operandOrig;
            T const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
        {
            operand = operandOrig;
            T const ret = alpaka::atomicCas(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, equals(operandOrig, ret));
            ALPAKA_CHECK(*success, equals(operand, reference));
        }
    }
}

template<typename TAcc, typename T, typename Sfinae = void>
class AtomicTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        if constexpr(std::is_unsigned_v<T>)
        {
            // atomicInc / atomicDec are implemented only for unsigned integer types
            testAtomicInc(acc, success, operandOrig);
            testAtomicDec(acc, success, operandOrig);
        }

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

template<typename TAcc, typename T>
class AtomicTestKernel<TAcc, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // These are not supported on float/double types
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)

template<typename TApi, typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    alpaka::AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
    T,
    std::enable_if_t<sizeof(T) != 4u && sizeof(T) != 8u>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuUniformCudaHipRt<TApi, TDim, TIdx> const& /* acc */,
        bool* success,
        T /* operandOrig */) const -> void
    {
        // All other types are not supported by CUDA/HIP atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};

#endif

template<typename TAcc, typename T>
struct TestAtomicOperations
{
    static auto testAtomicOperations() -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        AtomicTestKernel<TAcc, T> kernel;

        T value = static_cast<T>(32);
        REQUIRE(fixture(kernel, value));
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("atomicOperationsWorking", "[atomic]", TestAccs)
{
    using Acc = TestType;

    // According to the CUDA 12.1 Programming Guide, Section 7.14. Atomic Functions, an atomic function performs a
    // read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory.
    // Some operations require a compute capability of 5.0, 6.0, or higher; on older devices they can be emulated with
    // an atomicCAS loop.

    // According to SYCL 2020 rev. 7, Section 4.15.3. Atomic references, the template parameter T must be one of the
    // following types:
    //   - int, unsigned int,
    //   - long, unsigned long,
    //   - long long, unsigned long long,
    //   - float, or double.
    // In addition, the type T must satisfy one of the following conditions:
    //  - sizeof(T) == 4, or
    //  - sizeof(T) == 8 and the code containing the atomic_ref was submitted to a device that has aspect::atomic64.

    TestAtomicOperations<Acc, unsigned int>::testAtomicOperations();
    TestAtomicOperations<Acc, int>::testAtomicOperations();

    TestAtomicOperations<Acc, unsigned long>::testAtomicOperations();
    TestAtomicOperations<Acc, long>::testAtomicOperations();

    TestAtomicOperations<Acc, unsigned long long>::testAtomicOperations();
    TestAtomicOperations<Acc, long long>::testAtomicOperations();

    TestAtomicOperations<Acc, float>::testAtomicOperations();
    TestAtomicOperations<Acc, double>::testAtomicOperations();
}
