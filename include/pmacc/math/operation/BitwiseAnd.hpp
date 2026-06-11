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
