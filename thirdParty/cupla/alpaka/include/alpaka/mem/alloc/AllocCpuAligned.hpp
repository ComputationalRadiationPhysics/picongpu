/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/mem/alloc/Traits.hpp>

#include <alpaka/core/AlignedAlloc.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <algorithm>

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The allocator specifics.
        namespace alloc
        {
            //#############################################################################
            //! The CPU boost aligned allocator.
            //!
            //! \tparam TAlignment An integral constant containing the alignment.
            template<
                typename TAlignment>
            class AllocCpuAligned : public concepts::Implements<ConceptMemAlloc, AllocCpuAligned<TAlignment>>
            {
            };

            namespace traits
            {
                //#############################################################################
                //! The CPU boost aligned allocator memory allocation trait specialization.
                template<
                    typename T,
                    typename TAlignment>
                struct Alloc<
                    T,
                    AllocCpuAligned<TAlignment>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto alloc(
                        AllocCpuAligned<TAlignment> const & alloc,
                        std::size_t const & sizeElems)
                    -> T *
                    {
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                        // For CUDA host memory must be aligned to 4 kib to pin it with `cudaHostRegister`,
                        // this was described in older programming guides but was removed later.
                        // From testing with PIConGPU and cuda-memcheck we found out that the alignment is still required.
                        //
                        // For HIP the required alignment is the size of a cache line.
                        // https://rocm-developer-tools.github.io/HIP/group__Memory.html#gab8258f051e1a1f7385f794a15300e674
                        // To avoid issues with HIP(cuda) the alignment will be set also for HIP(clang)
                        // to 4kib.
                        // @todo evaluate requirements when the HIP ecosystem is more stable
                        constexpr size_t minAlignement = 4096;
#else
                        constexpr size_t minAlignement = TAlignment::value;
#endif
                        alpaka::ignore_unused(alloc);
                        return
                            reinterpret_cast<T *>(
                                core::alignedAlloc(std::max(TAlignment::value, minAlignement), sizeElems * sizeof(T)));
                    }
                };

                //#############################################################################
                //! The CPU boost aligned allocator memory free trait specialization.
                template<
                    typename T,
                    typename TAlignment>
                struct Free<
                    T,
                    AllocCpuAligned<TAlignment>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto free(
                        AllocCpuAligned<TAlignment> const & alloc,
                        T const * const ptr)
                    -> void
                    {
                        alpaka::ignore_unused(alloc);
                            core::alignedFree(
                                const_cast<void *>(
                                    reinterpret_cast<void const *>(ptr)));
                    }
                };
            }
        }
    }
}
