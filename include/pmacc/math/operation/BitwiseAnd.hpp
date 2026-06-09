/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2025 Tapish Narwal
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

namespace pmacc::math::operation
{
    //! Bitwise and
    struct BitwiseAnd
    {
        HDINLINE constexpr void operator()(auto& destination, auto const& source) const
        {
            destination &= source;
        }

        template<typename T_Worker>
        HDINLINE constexpr void operator()(T_Worker const&, auto& destination, auto const& source) const
        {
            destination &= source;
        }
    };

    namespace traits
    {
        template<>
        struct AlpakaAtomicOp<BitwiseAnd>
        {
            using type = alpaka::AtomicAnd;
        };

        /**
         * @brief The neutral element for BitwiseAnd is ~0 (all bits are 1).
         * @tparam T_Value The value type for which to get the neutral element.
         */
        template<typename T_Value>
        struct NeutralElement<BitwiseAnd, T_Value>
        {
            static constexpr T_Value value = T_Value(~0);
        };

    } // namespace traits

} // namespace pmacc::math::operation

namespace pmacc::mpi
{
    template<>
    HINLINE MPI_Op getMPI_Op<pmacc::math::operation::BitwiseAnd>()
    {
        return MPI_BAND;
    }
} // namespace pmacc::mpi
