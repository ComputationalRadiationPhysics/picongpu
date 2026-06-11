/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
