/* Copyright 2020 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/types.hpp"

#include <alpaka/core/Positioning.hpp>

namespace cupla
{
    inline namespace CUPLA_ACCELERATOR_NAMESPACE
    {
        inline namespace device
        {
            namespace hierarchy
            {
                //! hierarchy definitions for atomic operation
                using namespace ::alpaka::hierarchy;

            } // namespace hierarchy
        } // namespace device
    } // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
