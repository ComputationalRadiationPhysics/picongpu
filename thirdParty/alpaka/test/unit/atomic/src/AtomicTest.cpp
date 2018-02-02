/**
 * \file
 * Copyright 2016 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <boost/assert.hpp>
#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

//#############################################################################
class AtomicTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        T operandOrig) const
    -> void
    {
        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Add
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Add>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig + value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Sub
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Sub>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig - value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Min
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Min>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = (operandOrig < value) ? operandOrig : value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Max
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Max>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = (operandOrig > value) ? operandOrig : value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Exch
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Exch>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Inc
        {
            // \TODO: Check reset to 0 at 'value'.
            T operand = operandOrig;
            T const value = static_cast<T>(42);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Inc>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig + 1;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Dec
        {
            // \TODO: Check reset to 'value' at 0.
            T operand = operandOrig;
            T const value = static_cast<T>(42);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Dec>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig - 1;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::And
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::And>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig & value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Or
        {
            T operand = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Or>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig | value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Xor
        {
            T operand = operandOrig;
            T const value = operandOrig + static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Xor>(
                        acc,
                        &operand,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig ^ value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Cas with match
        {
            T operand = operandOrig;
            T const compare = operandOrig;
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Cas>(
                        acc,
                        &operand,
                        compare,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = value;
            BOOST_VERIFY(operand == reference);
        }

        //-----------------------------------------------------------------------------
        // alpaka::atomic::op::Cas without match
        {
            T operand = operandOrig;
            T const compare = operandOrig + static_cast<T>(1);
            T const value = static_cast<T>(4);
            T const ret =
                alpaka::atomic::atomicOp<
                    alpaka::atomic::op::Cas>(
                        acc,
                        &operand,
                        compare,
                        value);
            BOOST_VERIFY(operandOrig == ret);
            T const reference = operandOrig;
            BOOST_VERIFY(operand == reference);
        }
    }
};

BOOST_AUTO_TEST_SUITE(atomic)

//#############################################################################
template<
    typename TAcc,
    typename T,
    typename TSfinae = void>
struct TestAtomicOperations
{
    //-----------------------------------------------------------------------------
    static auto testAtomicOperations()
    -> void
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Size = alpaka::size::Size<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Size>::ones());

        AtomicTestKernel kernel;

        T value = static_cast<T>(32);
        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel,
                value));
    }
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
//#############################################################################
// NOTE: std::uint32_t is the only type supported by all atomic CUDA operations.
template<
    typename TDim,
    typename TSize,
    typename T>
struct TestAtomicOperations<
    alpaka::acc::AccGpuCudaRt<TDim, TSize>,
    T,
    typename std::enable_if<!std::is_same<std::uint32_t, T>::value>::type>
{
    //-----------------------------------------------------------------------------
    static auto testAtomicOperations()
    -> void
    {
        // All other types are not supported by all CUDA atomic operations.
    }
};
#endif

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    atomicOperationsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    // This test exceeds the maximum compilation time.
#if !ALPAKA_CI
    TestAtomicOperations<TAcc, std::int8_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::uint8_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::int16_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::uint16_t>::testAtomicOperations();
#endif
    TestAtomicOperations<TAcc, std::int32_t>::testAtomicOperations();
#if !ALPAKA_CI
    TestAtomicOperations<TAcc, std::uint32_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::int64_t>::testAtomicOperations();
    TestAtomicOperations<TAcc, std::uint64_t>::testAtomicOperations();
#endif
    // Not all atomic operations are possible with floating point values.
    //TestAtomicOperations<TAcc, float>::testAtomicOperations();
    //TestAtomicOperations<TAcc, double>::testAtomicOperations();
}

BOOST_AUTO_TEST_SUITE_END()
