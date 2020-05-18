/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/Unused.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

#include <climits>
#include <type_traits>

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicAdd(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Add>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig + value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicSub(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Sub>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig - value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicMin(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Min>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = (operandOrig < value) ? operandOrig : value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicMax(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Max>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = (operandOrig > value) ? operandOrig : value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicExch(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Exch>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicInc(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    // \TODO: Check reset to 0 at 'value'.
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(42);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Inc>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig + 1;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicDec(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    // \TODO: Check reset to 'value' at 0.
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(42);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Dec>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig - 1;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicAnd(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::And>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig & value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicOr(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Or>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig | value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicXor(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);
    operand = operandOrig;
    T const value = operandOrig + static_cast<T>(4);
    T const ret =
        alpaka::atomic::atomicOp<
            alpaka::atomic::op::Xor>(
                acc,
                &operand,
                value);
    ALPAKA_CHECK(*success, operandOrig == ret);
    T const reference = operandOrig ^ value;
    ALPAKA_CHECK(*success, operand == reference);
}

//-----------------------------------------------------------------------------
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TAcc,
    typename T>
ALPAKA_FN_ACC auto testAtomicCas(
    TAcc const & acc,
    bool * success,
    T operandOrig)
-> void
{
    auto && operand = alpaka::block::shared::st::allocVar<T, __COUNTER__>(acc);

    //-----------------------------------------------------------------------------
    // with match
    {
        operand = operandOrig;
        T const compare = operandOrig;
        T const value = static_cast<T>(4);
        T const ret =
            alpaka::atomic::atomicOp<
                alpaka::atomic::op::Cas>(
                    acc,
                    &operand,
                    compare,
                    value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        T const reference = value;
        ALPAKA_CHECK(*success, operand == reference);
    }

    //-----------------------------------------------------------------------------
    // without match
    {
        operand = operandOrig;
        T const compare = operandOrig + static_cast<T>(1);
        T const value = static_cast<T>(4);
        T const ret =
            alpaka::atomic::atomicOp<
                alpaka::atomic::op::Cas>(
                    acc,
                    &operand,
                    compare,
                    value);
        ALPAKA_CHECK(*success, operandOrig == ret);
        T const reference = operandOrig;
        ALPAKA_CHECK(*success, operand == reference);
    }
}

//#############################################################################
template<
    typename TAcc,
    typename T,
    typename Sfinae = void>
class AtomicTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        T operandOrig) const
    -> void
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
template<
    typename T,
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuUniformCudaHipRt<TDim, TIdx>,
    T>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuUniformCudaHipRt<TDim, TIdx> const & acc,
        bool * success,
        T operandOrig) const
    -> void
    {
        alpaka::ignore_unused(acc);
        alpaka::ignore_unused(success);
        alpaka::ignore_unused(operandOrig);
    }
};
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        int operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
// NOTE: unsigned int is the only type supported by all atomic CUDA operations.
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    unsigned int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        unsigned int operandOrig) const
    -> void
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
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    unsigned long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        unsigned long int operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
#if UINT_MAX == ULONG_MAX // LLP64
        testAtomicSub(acc, success, operandOrig);
#endif

#if ULONG_MAX == ULLONG_MAX // LP64
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#endif
#endif

        testAtomicExch(acc, success, operandOrig);

#if UINT_MAX == ULONG_MAX // LLP64
        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);
#endif

#if ULONG_MAX == ULLONG_MAX // LP64
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#endif
#endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    unsigned long long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        unsigned long long int operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        //testAtomicSub(acc, success, operandOrig);

#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#endif

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    float>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        float operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        //testAtomicSub(acc, success, operandOrig);

        // Not supported
        //testAtomicMin(acc, success, operandOrig);
        //testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

        // Not supported
        //testAtomicAnd(acc, success, operandOrig);
        //testAtomicOr(acc, success, operandOrig);
        //testAtomicXor(acc, success, operandOrig);

        // Not supported
        //testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    double>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        double operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        //testAtomicSub(acc, success, operandOrig);

        // Not supported
        //testAtomicMin(acc, success, operandOrig);
        //testAtomicMax(acc, success, operandOrig);

        // Not supported
        //testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

        // Not supported
        //testAtomicAnd(acc, success, operandOrig);
        //testAtomicOr(acc, success, operandOrig);
        //testAtomicXor(acc, success, operandOrig);

        // Not supported
        //testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx,
    typename T>
class AtomicTestKernel<
    alpaka::acc::AccGpuCudaRt<TDim, TIdx>,
    T,
    std::enable_if_t<
        !std::is_same<int, T>::value
        && !std::is_same<unsigned int, T>::value
        && !std::is_same<unsigned long int, T>::value
        && !std::is_same<unsigned long long int, T>::value
        && !std::is_same<float, T>::value
        && !std::is_same<double, T>::value
    >>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuCudaRt<TDim, TIdx> const & acc,
        bool * success,
        T operandOrig) const
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
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        int operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        testAtomicSub(acc, success, operandOrig);

        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
// NOTE: unsigned int is the only type supported by all atomic HIP operations.
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    unsigned int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        unsigned int operandOrig) const
    -> void
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
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    unsigned long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        unsigned long int operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
#if UINT_MAX == ULONG_MAX // LLP64
        testAtomicSub(acc, success, operandOrig);
#endif

#if ULONG_MAX == ULLONG_MAX // LP64
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#endif
#endif

        testAtomicExch(acc, success, operandOrig);

#if UINT_MAX == ULONG_MAX // LLP64
        testAtomicInc(acc, success, operandOrig);
        testAtomicDec(acc, success, operandOrig);
#endif

#if ULONG_MAX == ULLONG_MAX // LP64
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#endif
#endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    unsigned long long int>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        unsigned long long int operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        //testAtomicSub(acc, success, operandOrig);

#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicMin(acc, success, operandOrig);
        testAtomicMax(acc, success, operandOrig);
#endif

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
        testAtomicAnd(acc, success, operandOrig);
        testAtomicOr(acc, success, operandOrig);
        testAtomicXor(acc, success, operandOrig);
#endif

        testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    float>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        float operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        //testAtomicSub(acc, success, operandOrig);

        // Not supported
        //testAtomicMin(acc, success, operandOrig);
        //testAtomicMax(acc, success, operandOrig);

        testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

        // Not supported
        //testAtomicAnd(acc, success, operandOrig);
        //testAtomicOr(acc, success, operandOrig);
        //testAtomicXor(acc, success, operandOrig);

        // Not supported
        //testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    double>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        double operandOrig) const
    -> void
    {
        testAtomicAdd(acc, success, operandOrig);
        // Not supported
        //testAtomicSub(acc, success, operandOrig);

        // Not supported
        //testAtomicMin(acc, success, operandOrig);
        //testAtomicMax(acc, success, operandOrig);

        // Not supported
        //testAtomicExch(acc, success, operandOrig);

        // Not supported
        //testAtomicInc(acc, success, operandOrig);
        //testAtomicDec(acc, success, operandOrig);

        // Not supported
        //testAtomicAnd(acc, success, operandOrig);
        //testAtomicOr(acc, success, operandOrig);
        //testAtomicXor(acc, success, operandOrig);

        // Not supported
        //testAtomicCas(acc, success, operandOrig);
    }
};

//#############################################################################
template<
    typename TDim,
    typename TIdx,
    typename T>
class AtomicTestKernel<
    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
    T,
    std::enable_if_t<
        !std::is_same<int, T>::value
        && !std::is_same<unsigned int, T>::value
        && !std::is_same<unsigned long int, T>::value
        && !std::is_same<unsigned long long int, T>::value
        && !std::is_same<float, T>::value
        && !std::is_same<double, T>::value
    >>
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::acc::AccGpuHipRt<TDim, TIdx> const & acc,
        bool * success,
        T operandOrig) const
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
template<
    typename TAcc,
    typename T>
struct TestAtomicOperations
{
    //-----------------------------------------------------------------------------
    static auto testAtomicOperations()
    -> void
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        AtomicTestKernel<TAcc, T> kernel;

        T value = static_cast<T>(32);
        REQUIRE(fixture(kernel, value));
    }
};

using TestAccs = alpaka::test::acc::EnabledAccs<
    alpaka::dim::DimInt<1u>,
    std::size_t>;

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "atomicOperationsWorking", "[atomic]", TestAccs)
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
    //TestAtomicOperations<Acc, float>::testAtomicOperations();
    //TestAtomicOperations<Acc, double>::testAtomicOperations();
}
