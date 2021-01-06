/* Copyright 2020-2021 Rene Widera
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

#include "pmacc/types.hpp"

#include "pmacc/nvidia/atomic.hpp"

namespace pmacc
{
    namespace nvidia
    {
        namespace functors
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
                template<typename T_Acc, typename T_Dst, typename T_Src>
                HDINLINE void operator()(T_Acc const& acc, T_Dst& dst, T_Src const& src) const
                {
                    atomicOpNoRet<T_AlpakaOperation>(acc, &dst, src, T_AlpakaHierarchy{});
                }

                /** Execute atomic operation for pmacc::math::Vector */
                template<
                    typename T_Acc,
                    typename T_Type,
                    int T_dim,
                    typename T_DstAccessor,
                    typename T_DstNavigator,
                    template<typename, int>
                    class T_DstStorage,
                    typename T_SrcAccessor,
                    typename T_SrcNavigator,
                    template<typename, int>
                    class T_SrcStorage>
                HDINLINE void operator()(
                    T_Acc const& acc,
                    pmacc::math::Vector<T_Type, T_dim, T_DstAccessor, T_DstNavigator, T_DstStorage>& dst,
                    pmacc::math::Vector<T_Type, T_dim, T_SrcAccessor, T_SrcNavigator, T_SrcStorage> const& src) const
                {
                    for(int i = 0; i < T_dim; ++i)
                        atomicOpNoRet<T_AlpakaOperation>(acc, &dst[i], src[i], T_AlpakaHierarchy{});
                }
            };

        } // namespace functors
    } // namespace nvidia
} // namespace pmacc
