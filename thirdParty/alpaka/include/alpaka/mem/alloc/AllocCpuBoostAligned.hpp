/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/mem/alloc/Traits.hpp>

#include <alpaka/core/Common.hpp>

#include <boost/align.hpp>
#include <boost/core/ignore_unused.hpp>

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
                        boost::ignore_unused(alloc);
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
                        boost::ignore_unused(alloc);
                            boost::alignment::aligned_free(
                                const_cast<void *>(
                                    reinterpret_cast<void const *>(ptr)));
                    }
                };
            }
        }
    }
}
