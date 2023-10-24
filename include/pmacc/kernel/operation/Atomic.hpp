/* Copyright 2020-2023 Rene Widera
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

#include "pmacc/kernel/atomic.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace kernel
    {
        namespace operation
        {
            /** Addition of two values
             *
             * @tparam T_AlpakaOperation alpaka atomic operation [::alpaka::op]
             * @tparam T_AlpakaHierarchy alpaka atomic hierarchy [::alpaka::hierarchy]
             */
            template<typename T_AlpakaOperation, typename T_AlpakaHierarchy = ::alpaka::hierarchy::Grids>
            struct Atomic
            {
                /** Execute generic atomic operation */
                template<typename T_Worker, typename T_Dst, typename T_Src>
                HDINLINE void operator()(T_Worker const& worker, T_Dst& dst, T_Src const& src) const
                {
                    ::alpaka::atomicOp<T_AlpakaOperation>(worker.getAcc(), &dst, src, T_AlpakaHierarchy{});
                }

                /** Execute atomic operation for pmacc::math::Vector */
                template<
                    typename T_Worker,
                    typename T_Type,
                    uint32_t T_dim,
                    typename T_DstAccessor,
                    typename T_DstNavigator,
                    typename T_DstStorage,
                    typename T_SrcAccessor,
                    typename T_SrcNavigator,
                    typename T_SrcStorage>
                HDINLINE void operator()(
                    T_Worker const& worker,
                    pmacc::math::Vector<T_Type, T_dim, T_DstAccessor, T_DstNavigator, T_DstStorage>& dst,
                    pmacc::math::Vector<T_Type, T_dim, T_SrcAccessor, T_SrcNavigator, T_SrcStorage> const& src) const
                {
                    for(uint32_t i = 0; i < T_dim; ++i)
                        ::alpaka::atomicOp<T_AlpakaOperation>(worker.getAcc(), &dst[i], src[i], T_AlpakaHierarchy{});
                }
            };

        } // namespace operation
    } // namespace kernel
} // namespace pmacc
