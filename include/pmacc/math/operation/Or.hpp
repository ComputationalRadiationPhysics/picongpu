/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc::math::operation
{
    //! logical or
    struct Or
    {
        HDINLINE constexpr void operator()(uint32_t& destination, uint32_t const& source) const
        {
            destination = static_cast<uint32_t>(static_cast<bool>(destination) || static_cast<bool>(source));
        }

        template<typename T_Worker>
        HDINLINE constexpr void operator()(T_Worker const&, uint32_t& destination, uint32_t const& source) const
        {
            destination = static_cast<uint32_t>(static_cast<bool>(destination) || static_cast<bool>(source));
        }
    };
} // namespace pmacc::math::operation

namespace pmacc::mpi
{
    template<>
    HINLINE MPI_Op getMPI_Op<pmacc::math::operation::Or>()
    {
        return MPI_LOR;
    }
} // namespace pmacc::mpi
