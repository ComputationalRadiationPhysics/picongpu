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

namespace pmacc::math::operation::traits
{

    /**
     * @brief Trait to get the alpaka atomic operation for a pmacc math operation
     * @tparam T_Op The mathematical operation (e.g. pmacc::math::operation::Add)
     */
    template<typename T_Op>
    struct AlpakaAtomicOp;

    template<typename T_Op>
    using AlpakaAtomicOp_t = typename AlpakaAtomicOp<T_Op>::type;

    /**
     * @brief Trait to get the neutral element for a mathematical operation.
     * @tparam T_Op The mathematical operation (e.g. pmacc::math::operation::Add)
     * @tparam T_Value The value type for which to get the neutral element.
     */
    template<typename T_Op, typename T_Value>
    struct NeutralElement;

    template<typename T_Op, typename T_Value>
    inline constexpr T_Value NeutralElement_v = NeutralElement<T_Op, T_Value>::value;


} // namespace pmacc::math::operation::traits
