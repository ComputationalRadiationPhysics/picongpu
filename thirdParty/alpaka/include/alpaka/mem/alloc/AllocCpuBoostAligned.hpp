/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/mem/alloc/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <boost/align.hpp>

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
            class AllocCpuBoostAligned
            {
            public:
                using AllocBase = AllocCpuBoostAligned<TAlignment>;
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
                    AllocCpuBoostAligned<TAlignment>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto alloc(
                        AllocCpuBoostAligned<TAlignment> const & alloc,
                        std::size_t const & sizeElems)
                    -> T *
                    {
                        alpaka::ignore_unused(alloc);
                        return
                            reinterpret_cast<T *>(
                                boost::alignment::aligned_alloc(TAlignment::value, sizeElems * sizeof(T)));
                    }
                };

                //#############################################################################
                //! The CPU boost aligned allocator memory free trait specialization.
                template<
                    typename T,
                    typename TAlignment>
                struct Free<
                    T,
                    AllocCpuBoostAligned<TAlignment>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto free(
                        AllocCpuBoostAligned<TAlignment> const & alloc,
                        T const * const ptr)
                    -> void
                    {
                        alpaka::ignore_unused(alloc);
                            boost::alignment::aligned_free(
                                const_cast<void *>(
                                    reinterpret_cast<void const *>(ptr)));
                    }
                };
            }
        }
    }
}
