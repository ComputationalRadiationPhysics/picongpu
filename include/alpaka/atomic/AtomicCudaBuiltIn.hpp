/**
 * \file
 * Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The GPU CUDA accelerator atomic ops.
        //
        //  Atomics can used in the hierarchy level grids, blocks and threads.
        //  Atomics are not guaranteed to be save between devices
        class AtomicCudaBuiltIn
        {
        public:

            //-----------------------------------------------------------------------------
            AtomicCudaBuiltIn() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AtomicCudaBuiltIn(AtomicCudaBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY AtomicCudaBuiltIn(AtomicCudaBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(AtomicCudaBuiltIn const &) -> AtomicCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(AtomicCudaBuiltIn &&) -> AtomicCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicCudaBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The specializations to execute the requested atomic ops of the CUDA accelerator.
            // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions how to implement everything with CAS

            //-----------------------------------------------------------------------------
            // Add.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicCudaBuiltIn,
                float,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    float * const addr,
                    float const & value)
                -> float
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicCudaBuiltIn,
                double,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    double * const addr,
                    double const & value)
                -> double
                {
#if BOOST_ARCH_CUDA_DEVICE >= BOOST_VERSION_NUMBER(6, 0, 0)
                    return atomicAdd(addr, value);
#else
                    // Code from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

                    unsigned long long int * address_as_ull(reinterpret_cast<unsigned long long int *>(addr));
                    unsigned long long int old(*address_as_ull);
                    unsigned long long int assumed;
                    do
                    {
                        assumed = old;
                        old = atomicCAS(
                            address_as_ull,
                            assumed,
                            static_cast<unsigned long long>(__double_as_longlong(value + __longlong_as_double(static_cast<long long>(assumed)))));
                        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    }
                    while(assumed != old);
                    return __longlong_as_double(static_cast<long long>(old));
#endif
                }
            };

            //-----------------------------------------------------------------------------
            // Sub.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicSub(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicSub(addr, value);
                }
            };

            //-----------------------------------------------------------------------------
            // Min.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicMin(addr, value);
                }
            };*/

            //-----------------------------------------------------------------------------
            // Max.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicMax(addr, value);
                }
            };*/

            //-----------------------------------------------------------------------------
            // Exch.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicCudaBuiltIn,
                float,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    float * const addr,
                    float const & value)
                -> float
                {
                    return atomicExch(addr, value);
                }
            };

            //-----------------------------------------------------------------------------
            // Inc.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicInc(addr, value);
                }
            };

            //-----------------------------------------------------------------------------
            // Dec.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicDec(addr, value);
                }
            };

            //-----------------------------------------------------------------------------
            // And.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            /*template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicAnd(addr, value);
                }
            };*/

            //-----------------------------------------------------------------------------
            // Or.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicOr(addr, value);
                }
            };*/

            //-----------------------------------------------------------------------------
            // Xor.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicXor(addr, value);
                }
            };*/

            //-----------------------------------------------------------------------------
            // Cas.

            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicCudaBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    int * const addr,
                    int const & compare,
                    int const & value)
                -> int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicCudaBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & compare,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU CUDA accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicCudaBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicCudaBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & compare,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
        }
    }
}

#endif
