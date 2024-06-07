/* Copyright 2023-2024 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc::math::operation
{
    //! logical and
    struct And
    {
        HDINLINE void operator()(uint32_t& destination, uint32_t const& source) const
        {
            destination = static_cast<uint32_t>(static_cast<bool>(destination) && static_cast<bool>(source));
        }

        template<typename T_Worker>
        HDINLINE void operator()(T_Worker const&, uint32_t& destination, uint32_t const& source) const
        {
            destination = static_cast<uint32_t>(static_cast<bool>(destination) && static_cast<bool>(source));
        }
    };
} // namespace pmacc::math::operation
