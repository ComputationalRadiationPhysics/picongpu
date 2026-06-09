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
    //! Bitwise or
    struct BitwiseOr
    {
        HDINLINE constexpr void operator()(auto& destination, auto const& source) const
        {
            destination |= source;
        }

        template<typename T_Worker>
        HDINLINE constexpr void operator()(T_Worker const&, auto& destination, auto const& source) const
        {
            destination |= source;
        }
    };

    namespace traits
    {
        template<>
        struct AlpakaAtomicOp<BitwiseOr>
        {
            using type = alpaka::AtomicOr;
        };

        /**
         * @brief The neutral element for BitwiseOr is 0.
         * @tparam T_Value The value type for which to get the neutral element.
         */
        template<typename T_Value>
        struct NeutralElement<BitwiseOr, T_Value>
        {
            static constexpr T_Value value = T_Value(0);
        };

    } // namespace traits

} // namespace pmacc::math::operation

namespace pmacc::mpi
{
    template<>
    HINLINE MPI_Op getMPI_Op<pmacc::math::operation::BitwiseOr>()
    {
        return MPI_BOR;
    }
} // namespace pmacc::mpi
