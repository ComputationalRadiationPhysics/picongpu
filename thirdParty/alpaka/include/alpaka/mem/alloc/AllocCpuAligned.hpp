/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/AlignedAlloc.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/cpu/SysInfo.hpp"
#include "alpaka/mem/alloc/Traits.hpp"

#include <algorithm>

namespace alpaka
{
    //! The CPU boost aligned allocator.
    //!
    //! \tparam TAlignment An integral constant containing the alignment.
    template<typename TAlignment>
    class AllocCpuAligned : public concepts::Implements<ConceptMemAlloc, AllocCpuAligned<TAlignment>>
    {
    };

    namespace trait
    {
        //! The CPU boost aligned allocator memory allocation trait specialization.
        template<typename T, typename TAlignment>
        struct Malloc<T, AllocCpuAligned<TAlignment>>
        {
            ALPAKA_FN_HOST static auto malloc(
                AllocCpuAligned<TAlignment> const& /* alloc */,
                std::size_t const& sizeElems) -> T*
            {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                // For CUDA/HIP host memory must be aligned to 4 kib to pin it with `cudaHostRegister`,
                // this was described in older programming guides but was removed later.
                // From testing with PIConGPU and cuda-memcheck we found out that the alignment is still required.
                //
                // For HIP the required alignment is the size of a cache line.
                // https://rocm-developer-tools.github.io/HIP/group__Memory.html#gab8258f051e1a1f7385f794a15300e674
                // On most x86 systems the page size is 4KiB and on OpenPower 64KiB.
                // Page size can be tested on the terminal with: `getconf PAGE_SIZE`
                size_t minAlignement = std::max<size_t>(TAlignment::value, cpu::detail::getPageSize());
#else
                constexpr size_t minAlignement = TAlignment::value;
#endif
                return reinterpret_cast<T*>(core::alignedAlloc(minAlignement, sizeElems * sizeof(T)));
            }
        };

        //! The CPU boost aligned allocator memory free trait specialization.
        template<typename T, typename TAlignment>
        struct Free<T, AllocCpuAligned<TAlignment>>
        {
            ALPAKA_FN_HOST static auto free(AllocCpuAligned<TAlignment> const& /* alloc */, T const* const ptr) -> void
            {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                size_t minAlignement = std::max<size_t>(TAlignment::value, cpu::detail::getPageSize());
#else
                constexpr size_t minAlignement = TAlignment::value;
#endif
                core::alignedFree(minAlignement, const_cast<void*>(reinterpret_cast<void const*>(ptr)));
            }
        };
    } // namespace trait
} // namespace alpaka
