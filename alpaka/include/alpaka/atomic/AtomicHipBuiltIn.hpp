/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/core/Unused.hpp>
#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <climits>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The GPU HIP accelerator atomic ops.
        //
        //  Atomics can used in the hierarchy level grids, blocks and threads.
        //  Atomics are not guaranteed to be save between devices
        class AtomicHipBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            AtomicHipBuiltIn() = default;
            //-----------------------------------------------------------------------------
            __device__ AtomicHipBuiltIn(AtomicHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ AtomicHipBuiltIn(AtomicHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AtomicHipBuiltIn const &) -> AtomicHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AtomicHipBuiltIn &&) -> AtomicHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ALPAKA_FN_HOST_ACC ~AtomicHipBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The specializations to execute the requested atomic ops of the HIP accelerator.
            // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions how to implement everything with CAS

            //-----------------------------------------------------------------------------
            // Add.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicAdd(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
                    return atomicAdd(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
#endif
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                float,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    float * const addr,
                    float const & value)
                -> float
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                double,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    double * const addr,
                    double const & value)
                -> double
                {
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(6,0,0)
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
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------

                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicSub(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicSub(addr, value);
                }
            };

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicSub(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Sub, atomic::AtomicHipBuiltIn, unsigned long int> is only supported when sizeof(unsigned long int) == 4");
#endif
                }
            };

            //-----------------------------------------------------------------------------
            // Min.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------

                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicMin(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
# if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicMin(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
# else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Min, atomic::AtomicHipBuiltIn, unsigned long int> is only supported on sm >= 3.5");
# endif
#endif
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicMin(addr, value);
#else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Min, atomic::AtomicHipBuiltIn, unsigned long long int> is only supported on sm >= 3.5");
#endif
                }
            };
            //-----------------------------------------------------------------------------
            // Max.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicMax(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
# if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicMax(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
# else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Max, atomic::AtomicHipBuiltIn, unsigned long int> is only supported on sm >= 3.5");
# endif
#endif
                }
            };
          //-----------------------------------------------------------------------------
          //! The GPU HIP accelerator atomic operation.
          template<
            typename THierarchy>
          struct AtomicOp<
            op::Max,
            atomic::AtomicHipBuiltIn,
            unsigned long long int,
            THierarchy>
          {
            //-----------------------------------------------------------------------------
            __device__ static auto atomicOp(
              atomic::AtomicHipBuiltIn const &,
              unsigned long long int * const addr,
              unsigned long long int const & value)
              -> unsigned long long int
              {
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                return atomicMax(addr, value);
#else
                alpaka::ignore_unused(addr);
                alpaka::ignore_unused(value);
                static_assert(
                  meta::DependentFalseType<THierarchy>::value,
                  "atomicOp<op::Max, atomic::AtomicHipBuiltIn, unsigned long long int> is only supported on sm >= 3.5");
#endif
                    }
            };

            //-----------------------------------------------------------------------------
            // Exch.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicExch(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
                    return atomicExch(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
#endif
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                float,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
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
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicInc(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicInc(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Inc, atomic::AtomicHipBuiltIn, unsigned long int> is only supported when sizeof(unsigned long int) == 4");
#endif
                }
            };

            //-----------------------------------------------------------------------------
            // Dec.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicDec(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicDec(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Dec, atomic::AtomicHipBuiltIn, unsigned long int> is only supported when sizeof(unsigned long int) == 4");
#endif
                }
            };

            //-----------------------------------------------------------------------------
            // And.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicAnd(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
# if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicAnd(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
# else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::And, atomic::AtomicHipBuiltIn, unsigned long int> is only supported on sm >= 3.5");
# endif
#endif
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicAnd(addr, value);
#else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::And, atomic::AtomicHipBuiltIn, unsigned long long int> is only supported on sm >= 3.5");
#endif
                }
            };
            //-----------------------------------------------------------------------------
            // Or.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicOr(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
# if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicOr(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
# else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Or, atomic::AtomicHipBuiltIn, unsigned long int> is only supported on sm >= 3.5");
# endif
#endif
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicOr(addr, value);
#else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Or, atomic::AtomicHipBuiltIn, unsigned long long int> is only supported on sm >= 3.5");
#endif
                }
            };
            //-----------------------------------------------------------------------------
            // Xor.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicXor(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
# if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicXor(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(value));
# else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Xor, atomic::AtomicHipBuiltIn, unsigned long int> is only supported on sm >= 3.5");
# endif
#endif
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
#if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                    return atomicXor(addr, value);
#else
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<op::Xor, atomic::AtomicHipBuiltIn, unsigned long long int> is only supported on sm >= 3.5");
#endif
                }
            };

            //-----------------------------------------------------------------------------
            // Cas.

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & compare,
                    int const & value)
                -> int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & compare,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                unsigned long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long int * const addr,
                    unsigned long int const & compare,
                    unsigned long int const & value)
                -> unsigned long int
                {
#if UINT_MAX == ULONG_MAX // LLP64
                    return atomicCAS(
                        reinterpret_cast<unsigned int *>(addr),
                        static_cast<unsigned int>(compare),
                        static_cast<unsigned int>(value));
#else // ULONG_MAX == ULLONG_MAX LP64
                    return atomicCAS(
                        reinterpret_cast<unsigned long long int *>(addr),
                        static_cast<unsigned long long int>(compare),
                        static_cast<unsigned long long int>(value));
#endif
                }
            };

            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & compare,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicCAS(addr, compare, value);
                }
            };

            //#############################################################################
            //! The GPU HIP accelerator atomic operation.
            template<
                typename TOp,
                typename T,
                typename THierarchy>
            struct AtomicOp<
                TOp,
                atomic::AtomicHipBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    alpaka::ignore_unused(atomic);
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<TOp, atomic::AtomicHipBuiltIn, T>(atomic, addr, value) is not supported!");

                    return T();
                }
                //-----------------------------------------------------------------------------
                __device__ static auto atomicOp(
                    atomic::AtomicHipBuiltIn const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    alpaka::ignore_unused(atomic);
                    alpaka::ignore_unused(addr);
                    alpaka::ignore_unused(compare);
                    alpaka::ignore_unused(value);
                    static_assert(
                        meta::DependentFalseType<THierarchy>::value,
                        "atomicOp<TOp, atomic::AtomicHipBuiltIn, T>(atomic, addr, compare, value) is not supported!");

                    return T();
                }
            };
        }
    }
}

#endif
