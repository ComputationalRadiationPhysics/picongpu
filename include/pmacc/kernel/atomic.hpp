/* Copyright 2015-2022 Rene Widera, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/kernel/warp.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/types.hpp"

#include <alpaka/intrinsic/Traits.hpp>
#include <alpaka/warp/Traits.hpp>

#include <climits>
#include <type_traits>


namespace pmacc
{
    namespace kernel
    {
        namespace detail
        {
            /** optimized atomic operation without return value
             *
             * For some backends PMacc is using optimized intrinsics to perform this operation.
             *
             * @tparam T_Op atomic alpaka operation type
             * @tparam T_Worker lockstep worker type
             * @tparam T_Type value type
             * @tparam T_Hierarchy alpaka hierarchy type of the atomic operation
             */
            template<typename T_Op, typename T_Worker, typename T_Type, typename T_Hierarchy>
            struct AtomicOpNoRet
            {
                /** perform the atomic operation
                 *
                 * @param acc alpaka accelerator context
                 * @param ptr pointer to destination memory
                 * @param value input value
                 * @param hierarchy alpaka hierarchy scope for atomics
                 */
                DINLINE void operator()(
                    T_Worker const& worker,
                    T_Type* ptr,
                    T_Type const value,
                    T_Hierarchy const& hierarchy)
                {
                    ::alpaka::atomicOp<T_Op>(worker.getAcc(), ptr, value, hierarchy);
                }
            };

#if(ALPAKA_ACC_GPU_HIP_ENABLED == 1 && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 401)
            /** HIP backend specialization for atomic add float
             *
             * Uses the intrinsic atomicAddNoRet available for AMD gpus only.
             */
            template<typename T_Hierarchy, uint32_t T_numSuggestedWorkers, typename... T_AccArgs>
            struct AtomicOpNoRet<
                ::alpaka::AtomicAdd,
                lockstep::
                    Worker<alpaka::AccGpuUniformCudaHipRt<alpaka::ApiHipRt, T_AccArgs...>, T_numSuggestedWorkers>,
                float,
                T_Hierarchy>
            {
                DINLINE void operator()(
                    alpaka::AccGpuUniformCudaHipRt<alpaka::ApiHipRt, T_AccArgs...> const& worker,
                    float* ptr,
                    float const value,
                    T_Hierarchy const& hierarchy)
                {
                    ::atomicAddNoRet(ptr, value);
                }
            };
#endif

            template<typename T_Type, bool T_isKepler>
            struct AtomicAllInc
            {
                template<typename T_Acc, typename T_Hierarchy>
                HDINLINE T_Type operator()(const T_Acc& acc, T_Type* ptr, const T_Hierarchy& hierarchy)
                {
                    return ::alpaka::atomicOp<::alpaka::AtomicAdd>(acc, ptr, T_Type(1), hierarchy);
                }
            };

#if CUPLA_DEVICE_COMPILE == 1 && (BOOST_LANG_CUDA || BOOST_COMP_HIP)
            /**
             * Trait that returns whether an optimized version of AtomicAllInc
             * exists for Kepler architectures (and up)
             */
            template<typename T>
            struct AtomicAllIncIsOptimized
            {
                inline static constexpr bool value
                    = std::is_same_v<
                          T,
                          int> || std::is_same_v<T, unsigned int> || std::is_same_v<T, long long int> || std::is_same_v<T, unsigned long long int> || std::is_same_v<T, float>;
            };

            /**
             * AtomicAllInc for Kepler and up
             * Defaults to unoptimized version for unsupported types
             */
            template<typename T_Type, bool T_UseOptimized = AtomicAllIncIsOptimized<T_Type>::value>
            struct AtomicAllIncKepler : public AtomicAllInc<T_Type, false>
            {
            };

            /**
             * Optimized version
             *
             * This warp aggregated atomic increment implementation based on nvidia parallel forall example
             * http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
             * (author: Andrew Adinetz, date: October 1th, 2014)
             *
             */
            template<typename T_Type>
            struct AtomicAllIncKepler<T_Type, true>
            {
                template<typename T_Acc, typename T_Hierarchy>
                HDINLINE T_Type operator()(const T_Acc& acc, T_Type* ptr, const T_Hierarchy& hierarchy)
                {
                    const auto mask = alpaka::warp::activemask(acc);
                    const auto leader = alpaka::ffs(acc, static_cast<std::make_signed_t<decltype(mask)>>(mask)) - 1;

                    auto result = T_Type{};
                    const int laneId = getLaneId();
                    /* Get the start value for this warp */
                    if(laneId == leader)
                        result = ::alpaka::atomicOp<::alpaka::AtomicAdd>(
                            acc,
                            ptr,
                            static_cast<T_Type>(alpaka::popcount(acc, mask)),
                            hierarchy);
                    result = warpBroadcast(result, leader);
                    /* Add offset per thread */
                    return result
                        + static_cast<T_Type>(
                               alpaka::popcount(acc, mask & ((static_cast<decltype(mask)>(1u) << laneId) - 1u)));
                }
            };

            /**
             * Optimized version for int64.
             * As CUDA atomicAdd does not support int64 directly we just cast it
             * and call the uint64 implementation
             */
            template<>
            struct AtomicAllIncKepler<long long int, true>
            {
                template<typename T_Acc, typename T_Hierarchy>
                HDINLINE long long int operator()(
                    const T_Acc& acc,
                    long long int* ptr,
                    const T_Hierarchy&,
                    const T_Hierarchy& hierarchy)
                {
                    return static_cast<long long int>(AtomicAllIncKepler<unsigned long long int>()(
                        acc,
                        reinterpret_cast<unsigned long long int*>(ptr),
                        hierarchy));
                }
            };

            template<typename T_Type>
            struct AtomicAllInc<T_Type, true> : public AtomicAllIncKepler<T_Type>
            {
            };
#endif // CUPLA_DEVICE_COMPILE == 1

        } // namespace detail

        /** optimized atomic increment
         *
         * - only optimized if PTX ISA >=3.0
         * - this atomic uses warp aggregation to speedup the operation compared to cuda `atomicInc()`
         * - cuda `atomicAdd()` is used if the compute architecture does not support warp aggregation
         * - all participate threads must change the same pointer (ptr) else the result is unspecified
         *
         * @param ptr pointer to memory (must be the same address for all threads in a block)
         *
         */
        template<typename T, typename T_Worker, typename T_Hierarchy>
        HDINLINE T atomicAllInc(const T_Worker& worker, T* ptr, const T_Hierarchy& hierarchy)
        {
            return detail::AtomicAllInc<T, (PMACC_CUDA_ARCH >= 300 || BOOST_COMP_HIP)>()(
                worker.getAcc(),
                ptr,
                hierarchy);
        }

        template<typename T>
        HDINLINE T atomicAllInc(T* ptr)
        {
            /* Dirty hack to call an alpaka accelerator based function.
             * Members of the fakeAcc will be uninitialized and must not be accessed.
             *
             * The id provider for particles is the only code where atomicAllInc is used without an accelerator.
             * @todo remove the unsafe faked accelerator
             */
            pmacc::memory::Array<cupla::AccThreadSeq, 1> fakeAcc;
            return detail::AtomicAllInc<T, (PMACC_CUDA_ARCH >= 300 || BOOST_COMP_HIP)>()(
                fakeAcc[0],
                ptr,
                ::alpaka::hierarchy::Grids());
        }

        /** optimized atomic value exchange
         *
         * - only optimized if PTX ISA >=2.0
         * - this atomic uses warp vote function to speedup the operation
         *   compared to cuda `atomicExch()`
         * - cuda `atomicExch()` is used if the compute architecture not supports
         *   warps vote functions
         * - all participate threads must change the same
         *   pointer (ptr) and set the same value, else the
         *   result is unspecified
         *
         * @param ptr pointer to memory (must be the same address for all threads in a block)
         * @param value new value (must be the same for all threads in a block)
         */
        template<typename T_Type, typename T_Worker, typename T_Hierarchy>
        DINLINE void atomicAllExch(
            const T_Worker& worker,
            T_Type* ptr,
            const T_Type value,
            const T_Hierarchy& hierarchy)
        {
#if CUPLA_DEVICE_COMPILE == 1 && (BOOST_LANG_CUDA || BOOST_COMP_HIP)
            const auto mask = alpaka::warp::activemask(worker.getAcc());
            const auto leader
                = alpaka::ffs(worker.getAcc(), static_cast<std::make_signed_t<decltype(mask)>>(mask)) - 1;
            if(getLaneId() == leader)
#endif
                ::alpaka::atomicOp<::alpaka::AtomicExch>(worker.getAcc(), ptr, value, hierarchy);
        }

        /** optimized atomic operation without return value
         *
         * Executes an alpaka atomic operation but without giving the old value back.
         * For some backends PMacc is using optimized intrinsics to perform this operation.
         *
         * @tparam T_Op atomic alpaka operation type
         * @tparam T_Worker lockstep worker type
         * @tparam T_Type value type
         * @tparam T_Hierarchy alpaka hierarchy type of the atomic operation
         * @param worker lockstep worker
         * @param ptr pointer to memory
         * @param value input value
         * @param hierarchy alpaka hierarchy scope for atomics
         */
        template<typename T_Op, typename T_Worker, typename T_Type, typename T_Hierarchy>
        DINLINE void atomicOpNoRet(
            T_Worker const& worker,
            T_Type* ptr,
            T_Type const value,
            T_Hierarchy const& hierarchy)
        {
            return detail::AtomicOpNoRet<T_Op, T_Worker, T_Type, T_Hierarchy>{}(worker, ptr, value, hierarchy);
        }
    } // namespace kernel
} // namespace pmacc
