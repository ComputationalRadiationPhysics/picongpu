/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuUniformCudaHipRt.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/exec/UniformElements.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/wait/Traits.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <iostream>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    template<typename TApi>
    class DevUniformCudaHipRt;

    namespace detail
    {
        template<typename TElem, typename TExtent, typename TPitchBytes>
        struct FillKernelND
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                TElem* ptr,
                TElem value,
                TExtent extent,
                TPitchBytes pitchBytes) const
            {
                for(auto const& idx : alpaka::uniformElementsND(acc, extent))
                {
                    // The host code checks that the pitches are a multiple of TElem's alignment.
                    std::uintptr_t offsetBytes = static_cast<std::uintptr_t>((pitchBytes * idx).sum());
                    TElem* elem = reinterpret_cast<TElem*>(
                        __builtin_assume_aligned(reinterpret_cast<std::uint8_t*>(ptr) + offsetBytes, alignof(TElem)));

                    // Write value at element address
                    *elem = value;
                }
            }
        };

        template<typename TElem>
        struct FillKernel0D
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()([[maybe_unused]] TAcc const& acc, TElem* ptr, TElem value) const
            {
                // A zero-dimensional buffer always has a single element.
                *ptr = value;
            }
        };


    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TApi>
        struct CreateTaskFill<TDim, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TViewFwd, typename TValue>
            ALPAKA_FN_HOST static auto createTaskFill(TViewFwd&& view, TValue const& value, TExtent const& extent)
            {
                using View = std::remove_reference_t<TViewFwd>;
                using Idx = alpaka::Idx<View>;
                using Acc = AccGpuUniformCudaHipRt<TApi, TDim, Idx>;
                using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
                using Vec = alpaka::Vec<TDim, Idx>;
                using Elem = alpaka::Elem<View>;
                static_assert(
                    std::is_trivially_copyable_v<Elem>,
                    "Only trivially copyable types are supported for fill");

                if constexpr(TDim::value == 0)
                {
                    // A zero-dimensional buffer always has a single element.
                    WorkDiv grid{Vec{}, Vec{}, Vec{}};
                    return alpaka::createTaskKernel<Acc>(
                        grid,
                        alpaka::detail::FillKernel0D<Elem>{},
                        std::data(view),
                        value);
                }
                else
                {
                    // TODO: compute an efficient work division.
                    Vec const elements = Vec::ones();
                    Vec threads = Vec::ones();
                    threads.x() = 64;
                    Vec const blocks = Vec::ones();
                    WorkDiv grid = WorkDiv(blocks, threads, elements);

                    // Check that the pitches are a multiple of Elem's alignment.
                    auto pitches = getPitchesInBytes(view);
                    for([[maybe_unused]] auto pitch : pitches)
                    {
                        ALPAKA_ASSERT(static_cast<std::size_t>(pitch) % alignof(Elem) == 0);
                    }
                    return alpaka::createTaskKernel<Acc>(
                        grid,
                        alpaka::detail::FillKernelND<Elem, TExtent, Vec>{},
                        std::data(view),
                        value,
                        extent,
                        pitches);
                }
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
