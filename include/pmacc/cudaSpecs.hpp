/* Copyright 2015-2023 Heiko Burau
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

#include <cstdint>

#include "pmacc/math/vector/compile-time/Size_t.hpp"

namespace pmacc
{
    namespace cudaSpecs
    {
        /* Various hardware specific numerical limits taken from the
         * *CUDA C Programming Guide* Section: G.1. Features and Technical Specifications.
         *
         * Valid for sm_2.x - sm_5.3
         */

        /** maximum number of threads per block */
        constexpr uint32_t maxNumThreadsPerBlock = 1024;

        /** maximum number of threads per axis of a block */
        using MaxNumThreadsPerBlockDim = math::CT::Size_t<1024, 1024, 64>;

    } // namespace cudaSpecs
} // namespace pmacc
