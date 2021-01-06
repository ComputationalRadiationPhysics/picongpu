/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "CartBuffer.hpp"
#include "../allocator/compile-time/SharedMemAllocator.hpp"

namespace pmacc
{
    namespace container
    {
        namespace CT
        {
            /* typedef version of container::CT::CartBuffer for shared mem on a GPU inside a cupla kernel.
             * \param uid If two containers in one kernel have the same Type and Size,
             * uid has to be different. This is due to a nvcc bug.
             */
            template<typename Type, typename Size, int uid = 0>
            using SharedBuffer = CT::
                CartBuffer<Type, Size, allocator::CT::SharedMemAllocator<Type, Size, Size::dim, uid>, void, void>;

        } // namespace CT
    } // namespace container
} // namespace pmacc
