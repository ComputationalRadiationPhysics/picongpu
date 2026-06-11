/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                    typename T_DstStorage,
                    typename T_SrcStorage>
                HDINLINE void operator()(
                    T_Worker const& worker,
                    pmacc::math::Vector<T_Type, T_dim, T_DstStorage>& dst,
                    pmacc::math::Vector<T_Type, T_dim, T_SrcStorage> const& src) const
                {
                    for(uint32_t i = 0; i < T_dim; ++i)
                        ::alpaka::atomicOp<T_AlpakaOperation>(worker.getAcc(), &dst[i], src[i], T_AlpakaHierarchy{});
                }
            };

        } // namespace operation
    } // namespace kernel
} // namespace pmacc
