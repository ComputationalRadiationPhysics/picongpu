/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <climits>
#include <type_traits>

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicAdd(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicSub(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicMin(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicMax(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicExch(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicInc(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicDec(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicAnd(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicOr(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
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
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
    {
        operand = operandOrig;
        T const ret = alpaka::atomicXor(acc, &operand, value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicCas(TAcc const& acc, bool* success, T operandOrig) -> void
{
    T const value = static_cast<T>(4);
    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);

    //-----------------------------------------------------------------------------
    // with match
    {
        T const compare = operandOrig;
        T const reference = value;
        {
            operand = operandOrig;
            T const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, operandOrig == ret);
            ALPAKA_CHECK(*success, operand == reference);
        }
        {
            operand = operandOrig;
            T const ret = alpaka::atomicCas(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, operandOrig == ret);
            ALPAKA_CHECK(*success, operand == reference);
        }
    }

    //-----------------------------------------------------------------------------
    // without match
    {
        T const compare = static_cast<T>(operandOrig + static_cast<T>(1));
        T const reference = operandOrig;
        {
            operand = operandOrig;
            T const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, operandOrig == ret);
            ALPAKA_CHECK(*success, operand == reference);
        }
        {
            operand = operandOrig;
            T const ret = alpaka::atomicCas(acc, &operand, compare, value);
            ALPAKA_CHECK(*success, operandOrig == ret);
            ALPAKA_CHECK(*success, operand == reference);
        }
    }
}

//#############################################################################
template<typename TAcc, typename T, typename Sfinae = void>
class AtomicTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
//#############################################################################
// Skip all atomic tests for the unified CUDA/HIP backend.
// CUDA and HIP atomics will be tested separate.
template<typename T, typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuUniformCudaHipRt<TDim, TIdx>, T>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuUniformCudaHipRt<TDim, TIdx> const& acc, bool* success, T operandOrig)
        const -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(success);
        alpaka::ignore_unused(operandOrig);
    }
};
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, int operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
// NOTE: unsigned int is the only type supported by all atomic CUDA operations.
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, unsigned int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, unsigned int operandOrig)
        const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, unsigned long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuCudaRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicSub(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#        endif
#    endif

        testAtomicExch(acc, success, operandOrig);

#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#        endif
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, unsigned long long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuCudaRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#    endif

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, float>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, float operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuCudaRt<TDim, TIdx>, double>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, double operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        // Not supported
        // testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    alpaka::AccGpuCudaRt<TDim, TIdx>,
    T,
    std::enable_if_t<
        !std::is_same<int, T>::value && !std::is_same<unsigned int, T>::value
        && !std::is_same<unsigned long int, T>::value && !std::is_same<unsigned long long int, T>::value
        && !std::is_same<float, T>::value && !std::is_same<double, T>::value>>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc, bool* success, T operandOrig) const
        -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(operandOrig);

        // All other types are not supported by CUDA atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, int operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
// NOTE: unsigned int is the only type supported by all atomic HIP operations.
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, unsigned int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, unsigned int operandOrig)
        const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, unsigned long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuHipRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicSub(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#        endif
#    endif

        testAtomicExch(acc, success, operandOrig);

#    if UINT_MAX == ULONG_MAX // LLP64
        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);
#    endif

#    if ULONG_MAX == ULLONG_MAX // LP64
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#        endif
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, unsigned long long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuHipRt<TDim, TIdx> const& acc,
        bool* success,
        unsigned long long int operandOrig) const -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#    endif

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

#    if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#    endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, float>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, float operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx>
class AtomicTestKernel<alpaka::AccGpuHipRt<TDim, TIdx>, double>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, double operandOrig) const
        -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        // testAtomicSub(acc, success, operandOrig);

        // Not supported
        // testAtomicMin(acc, success, operandOrig);
        // testAtomicMax(acc, success, operandOrig);

        // Not supported
        // testAtomicExch(acc, success, operandOrig);

        // Not supported
        // testAtomicInc(acc, success, operandOrig);
        // testAtomicDec(acc, success, operandOrig);

        // Not supported
        // testAtomicAnd(acc, success, operandOrig);
        // testAtomicOr(acc, success, operandOrig);
        // testAtomicXor(acc, success, operandOrig);

        // Not supported
        // testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    alpaka::AccGpuHipRt<TDim, TIdx>,
    T,
    std::enable_if_t<
        !std::is_same<int, T>::value && !std::is_same<unsigned int, T>::value
        && !std::is_same<unsigned long int, T>::value && !std::is_same<unsigned long long int, T>::value
        && !std::is_same<float, T>::value && !std::is_same<double, T>::value>>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuHipRt<TDim, TIdx> const& acc, bool* success, T operandOrig) const
        -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(operandOrig);

        // All other types are not supported by HIP atomic operations.
        ALPAKA_CHECK(*success, true);
    }
};
#endif


//#############################################################################
template<typename TAcc, typename T>
struct TestAtomicOperations
{
    //-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("atomicOperationsWorking", "[atomic]", TestAccs)
{
    using Acc = TestType;
    TestAtomicOperations<Acc, unsigned char>::testAtomicOperations();
    TestAtomicOperations<Acc, char>::testAtomicOperations();
    TestAtomicOperations<Acc, unsigned short>::testAtomicOperations();
    TestAtomicOperations<Acc, short>::testAtomicOperations();

    TestAtomicOperations<Acc, unsigned int>::testAtomicOperations();
    TestAtomicOperations<Acc, int>::testAtomicOperations();

    TestAtomicOperations<Acc, unsigned long>::testAtomicOperations();
    TestAtomicOperations<Acc, long>::testAtomicOperations();
    TestAtomicOperations<Acc, unsigned long long>::testAtomicOperations();
    TestAtomicOperations<Acc, long long>::testAtomicOperations();

    // Not all atomic operations are possible with floating point values.
    // TestAtomicOperations<Acc, float>::testAtomicOperations();
    // TestAtomicOperations<Acc, double>::testAtomicOperations();
}
