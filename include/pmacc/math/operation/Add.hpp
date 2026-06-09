/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2013-2024 Heiko Burau, Rene Widera, Benjamin Worpitz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
