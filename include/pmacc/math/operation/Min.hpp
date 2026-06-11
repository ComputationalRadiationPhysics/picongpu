/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/operation/traits.hpp"
#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        namespace operation
        {
            struct Min
            {
                template<typename Dst, typename Src>
                HDINLINE constexpr void operator()(Dst& dst, Src const& src) const
                {
                    dst = pmacc::math::min(dst, src);
                }

                template<typename Dst, typename Src, typename T_Worker>
                HDINLINE constexpr void operator()(T_Worker const& worker, Dst& dst, Src const& src) const
                {
                    dst = alpaka::math::min(worker.getAcc(), dst, src);
                }
            };

            namespace traits
            {
                template<>
                struct AlpakaAtomicOp<Min>
                {
                    using type = alpaka::AtomicMin;
                };

                /**
                 * @brief The neutral element for Min is the maximum representable number.
                 * @tparam T_Value The value type for which to get the neutral element.
                 */
                template<typename T_Value>
                struct NeutralElement<Min, T_Value>
                {
                    static constexpr T_Value value = T_Value(std::numeric_limits<T_Value>::max());
                };

            } // namespace traits

        } // namespace operation
    } // namespace math
} // namespace pmacc

namespace pmacc
{
    namespace mpi
    {
        template<>
        HINLINE MPI_Op getMPI_Op<pmacc::math::operation::Min>()
        {
            return MPI_MIN;
        }
    } // namespace mpi
} // namespace pmacc
