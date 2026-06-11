/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
