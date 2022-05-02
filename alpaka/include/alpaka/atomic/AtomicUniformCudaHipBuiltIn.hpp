/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/atomic/Op.hpp>
#    include <alpaka/atomic/Traits.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Unreachable.hpp>

#    include <limits>

namespace alpaka
{
    //! The GPU CUDA/HIP accelerator atomic ops.
    //
    //  Atomics can be used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be save between devices.
    class AtomicUniformCudaHipBuiltIn
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        //! The specializations to execute the requested atomic ops of the CUDA/HIP accelerator.
        // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions how to implement everything with
        // CAS

        // Add.

        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicAdd(addr, value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicAdd(addr, value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long int* const addr,
                unsigned long int const& value) -> unsigned long int
            {
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicAdd(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                {
                    return ::atomicAdd(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));
                }

                ALPAKA_UNREACHABLE(0ul);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long long int* const addr,
                unsigned long long int const& value) -> unsigned long long int
            {
                return ::atomicAdd(addr, value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformCudaHipBuiltIn, float, THierarchy>
        {
            //
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, float* const addr, float const& value)
                -> float
            {
                return ::atomicAdd(addr, value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformCudaHipBuiltIn, double, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                double* const addr,
                double const& value) -> double
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(6, 0, 0)
                return ::atomicAdd(addr, value);
#        else
                // Code from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

                unsigned long long int* address_as_ull(reinterpret_cast<unsigned long long int*>(addr));
                unsigned long long int old(*address_as_ull);
                unsigned long long int assumed;
                do
                {
                    assumed = old;
                    old = ::atomicCAS(
                        address_as_ull,
                        assumed,
                        static_cast<unsigned long long>(
                            __double_as_longlong(value + __longlong_as_double(static_cast<long long>(assumed)))));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                } while(assumed != old);
                return __longlong_as_double(static_cast<long long>(old));
#        endif
            }
        };

        // Sub.

        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicSub(addr, value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicSub(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const& ctx,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicSub(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return alpaka::atomicAdd(ctx, addr, -value);

                ALPAKA_UNREACHABLE(0ul);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long long int* const addr,
                unsigned long long int const& value) -> unsigned long long int
            {
                return ::atomicAdd(addr, -value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformCudaHipBuiltIn, float, THierarchy>
        {
            //
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, float* const addr, float const& value)
                -> float
            {
                return ::atomicAdd(addr, -value);
            }
        };
        //! The GPU CUDA/HIP accelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformCudaHipBuiltIn, double, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const& ctx,
                double* const addr,
                double const& value) -> double
            {
                return alpaka::atomicAdd(ctx, addr, -value);
            }
        };

        // Min.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicMin(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicMin(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicMin(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicMin(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, unsigned long int> is only supported on sm "
                    ">= 3.5");
                return 0ul;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long long int* const addr,
                [[maybe_unused]] unsigned long long int const& value) -> unsigned long long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                return ::atomicMin(addr, value);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, unsigned long long int> is only supported on sm "
                    ">= 3.5");
                return 0ull;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, float, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, float* const addr, float const& value)
                -> float
            {
                int* address_as_i(reinterpret_cast<int*>(addr));
                int old(*address_as_i);
                int assumed;
                do
                {
                    assumed = old;
                    old = ::atomicCAS(address_as_i, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
                } while(assumed != old);
                return __int_as_float(old);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformCudaHipBuiltIn, double, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                double* const addr,
                double const& value) -> double
            {
                unsigned long long int* address_as_ull(reinterpret_cast<unsigned long long int*>(addr));
                unsigned long long int old(*address_as_ull);
                unsigned long long int assumed;
                do
                {
                    assumed = old;
                    old = ::atomicCAS(
                        address_as_ull,
                        assumed,
                        static_cast<unsigned long long int>(__double_as_longlong(
                            fmin(value, __longlong_as_double(static_cast<long long int>(assumed))))));
                } while(assumed != old);
                return __longlong_as_double(static_cast<long long int>(old));
            }
        };

        // Max.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicMax(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicMax(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicMax(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicMax(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, unsigned long int> is only supported on sm >= "
                    "3.5");
                return 0ul;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long long int* const addr,
                [[maybe_unused]] unsigned long long int const& value) -> unsigned long long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                return ::atomicMax(addr, value);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, unsigned long long int> is only supported on sm "
                    ">= 3.5");
                return 0ull;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, float, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, float* const addr, float const& value)
                -> float
            {
                int* address_as_i(reinterpret_cast<int*>(addr));
                int old(*address_as_i);
                int assumed;
                do
                {
                    assumed = old;
                    old = ::atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
                } while(assumed != old);
                return __int_as_float(old);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformCudaHipBuiltIn, double, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                double* const addr,
                double const& value) -> double
            {
                unsigned long long int* address_as_ull(reinterpret_cast<unsigned long long int*>(addr));
                unsigned long long int old(*address_as_ull);
                unsigned long long int assumed;
                do
                {
                    assumed = old;
                    old = ::atomicCAS(
                        address_as_ull,
                        assumed,
                        static_cast<unsigned long long int>(__double_as_longlong(
                            fmax(value, __longlong_as_double(static_cast<long long int>(assumed))))));
                } while(assumed != old);
                return __longlong_as_double(static_cast<long long int>(old));
            }
        };

        // Exch.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicExch(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicExch(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long int* const addr,
                unsigned long int const& value) -> unsigned long int
            {
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicExch(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicExch(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long long int* const addr,
                unsigned long long int const& value) -> unsigned long long int
            {
                return ::atomicExch(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformCudaHipBuiltIn, float, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, float* const addr, float const& value)
                -> float
            {
                return ::atomicExch(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformCudaHipBuiltIn, double, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                double* const addr,
                double const& value) -> double
            {
                return __longlong_as_double(static_cast<long long>(::atomicExch(
                    reinterpret_cast<unsigned long long*>(addr),
                    static_cast<unsigned long long>(__double_as_longlong(value)))));
            }
        };

        // Inc.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                int old(*addr);
                int assumed, eval;
                do
                {
                    assumed = old;
                    eval = assumed >= value ? 0 : assumed + 1;
                    old = ::atomicCAS(addr, assumed, eval);
                } while(assumed != old);
                return old;
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicInc(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicInc(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                {
                    unsigned long long int* address_as_u = reinterpret_cast<unsigned long long int*>(addr);
                    unsigned long long int old(*address_as_u);
                    unsigned long long int assumed, eval;
                    do
                    {
                        assumed = old;
                        eval = assumed >= static_cast<unsigned long long int>(value) ? 0 : assumed + 1;
                        old = ::atomicCAS(address_as_u, assumed, eval);
                    } while(assumed != old);
                    return static_cast<unsigned long int>(old);
                }

                ALPAKA_UNREACHABLE(0ul);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long long int* const addr,
                unsigned long long int const& value) -> unsigned long long int
            {
                unsigned long long int old(*addr);
                unsigned long long int assumed, eval;
                do
                {
                    assumed = old;
                    eval = assumed >= value ? 0 : assumed + 1;
                    old = ::atomicCAS(addr, assumed, eval);
                } while(assumed != old);
                return old;
            }
        };

        // Dec.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                int old(*addr);
                int assumed, eval;
                do
                {
                    assumed = old;
                    eval = assumed == 0 || assumed > value ? value : assumed - 1;
                    old = ::atomicCAS(addr, assumed, eval);
                } while(assumed != old);
                return old;
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicDec(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicDec(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                {
                    unsigned long long int* address_as_u = reinterpret_cast<unsigned long long int*>(addr);
                    unsigned long long int old(*address_as_u);
                    unsigned long long int assumed, eval;
                    do
                    {
                        assumed = old;
                        eval = assumed == 0 || assumed > static_cast<unsigned long long int>(value)
                            ? static_cast<unsigned long long int>(value)
                            : assumed - 1;
                        old = ::atomicCAS(address_as_u, assumed, eval);
                    } while(assumed != old);
                    return static_cast<unsigned long int>(old);
                }

                ALPAKA_UNREACHABLE(0ul);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long long int* const addr,
                unsigned long long int const& value) -> unsigned long long int
            {
                unsigned long long int old(*addr);
                unsigned long long int assumed, eval;
                do
                {
                    assumed = old;
                    eval = assumed == 0 || assumed > value ? value : assumed - 1;
                    old = ::atomicCAS(addr, assumed, eval);
                } while(assumed != old);
                return old;
            }
        };

        // And.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicAnd(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicAnd(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicAnd(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicAnd(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicAnd, AtomicUniformCudaHipBuiltIn, unsigned long int> is only supported on sm >= "
                    "3.5");
                return 0ul;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long long int* const addr,
                [[maybe_unused]] unsigned long long int const& value) -> unsigned long long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                return ::atomicAnd(addr, value);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicAnd, AtomicUniformCudaHipBuiltIn, unsigned long long int> is only supported on sm "
                    ">= 3.5");
                return 0ull;
#        endif
            }
        };

        // Or.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicOr(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicOr(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicOr(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicOr(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicOr, AtomicUniformCudaHipBuiltIn, unsigned long int> is only supported on sm >= "
                    "3.5");
                return 0ul;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long long int* const addr,
                [[maybe_unused]] unsigned long long int const& value) -> unsigned long long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                return ::atomicOr(addr, value);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicOr, AtomicUniformCudaHipBuiltIn, unsigned long long int> is only supported on sm "
                    ">= 3.5");
                return 0ull;
#        endif
            }
        };

        // Xor.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(AtomicUniformCudaHipBuiltIn const&, int* const addr, int const& value)
                -> int
            {
                return ::atomicXor(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicXor(addr, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long int* const addr,
                [[maybe_unused]] unsigned long int const& value) -> unsigned long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicXor(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicXor(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicXor, AtomicUniformCudaHipBuiltIn, unsigned long int> is only supported on sm >= "
                    "3.5");
                return 0ul;
#        endif
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                [[maybe_unused]] unsigned long long int* const addr,
                [[maybe_unused]] unsigned long long int const& value) -> unsigned long long int
            {
#        if BOOST_ARCH_PTX >= BOOST_VERSION_NUMBER(3, 5, 0)
                return ::atomicXor(addr, value);
#        else
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<AtomicXor, AtomicUniformCudaHipBuiltIn, unsigned long long int> is only supported on sm "
                    ">= 3.5");
                return 0ull;
#        endif
            }
        };

        // Cas.

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                int* const addr,
                int const& compare,
                int const& value) -> int
            {
                return ::atomicCAS(addr, compare, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, unsigned int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned int* const addr,
                unsigned int const& compare,
                unsigned int const& value) -> unsigned int
            {
                return ::atomicCAS(addr, compare, value);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, unsigned long int, THierarchy>
        {
            //
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long int* const addr,
                unsigned long int const& compare,
                unsigned long int const& value) -> unsigned long int
            {
                // LLP64
                if constexpr(std::numeric_limits<unsigned int>::max() == std::numeric_limits<unsigned long>::max())
                    return ::atomicCAS(
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(compare),
                        static_cast<unsigned int>(value));
                else // LP64
                    return ::atomicCAS(
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(compare),
                        static_cast<unsigned long long int>(value));

                ALPAKA_UNREACHABLE(0ul);
            }
        };
        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, unsigned long long int, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const&,
                unsigned long long int* const addr,
                unsigned long long int const& compare,
                unsigned long long int const& value) -> unsigned long long int
            {
                return ::atomicCAS(addr, compare, value);
            }
        };

        //! The GPU CUDA/HIPaccelerator atomic operation.
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicUniformCudaHipBuiltIn, T, THierarchy>
        {
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const& /* atomic */,
                T* const /* addr */,
                T const& /* value */) -> T
            {
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<TOp, AtomicUniformCudaHipBuiltIn, T>(atomic, addr, value) is not supported!");
                return T();
            }
            __device__ static auto atomicOp(
                AtomicUniformCudaHipBuiltIn const& /* atomic */,
                T* const /* addr */,
                T const& /* compare */,
                T const& /* value */) -> T
            {
                static_assert(
                    !sizeof(THierarchy),
                    "atomicOp<TOp, AtomicUniformCudaHipBuiltIn, T>(atomic, addr, compare, value) is not supported!");
                return T();
            }
        };
    } // namespace trait

#    endif

} // namespace alpaka

#endif
