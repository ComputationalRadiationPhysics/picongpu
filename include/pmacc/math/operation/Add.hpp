/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/math/operation/traits.hpp"
#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        namespace operation
        {
            struct Add
            {
                template<typename Dst, typename Src>
                HDINLINE constexpr void operator()(Dst& dst, Src const& src) const
                {
                    dst += src;
                }

                template<typename Dst, typename Src, typename T_Worker>
                HDINLINE constexpr void operator()(T_Worker const&, Dst& dst, Src const& src) const
                {
                    dst += src;
                }
            };

            namespace traits
            {
                template<>
                struct AlpakaAtomicOp<Add>
                {
                    using type = alpaka::AtomicAdd;
                };

                /**
                 * @brief The neutral element for addition is 0.
                 * @tparam T_Value The value type for which to get the neutral element.
                 */
                template<typename T_Value>
                struct NeutralElement<Add, T_Value>
                {
                    static constexpr T_Value value = T_Value(0);
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
        HINLINE MPI_Op getMPI_Op<pmacc::math::operation::Add>()
        {
            return MPI_SUM;
        }
    } // namespace mpi
} // namespace pmacc
