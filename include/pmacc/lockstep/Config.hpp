/* Copyright 2017-2023 Rene Widera
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

namespace pmacc
{
    namespace lockstep
    {
        /** describe a constant index domain
         *
         * describe the size of the index domain and the number of workers to operate on a lockstep domain
         *
         * @tparam T_domainSize number of indices in the domain
         * @tparam T_numWorkers number of worker working on @p T_domainSize
         * @tparam T_simdSize SIMD width
         */
        template<uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        struct Config
        {
            /** number of indices within the domain */
            static constexpr uint32_t domainSize = T_domainSize;
            /** number of worker (threads) working on @p domainSize */
            static constexpr uint32_t numWorkers = T_numWorkers;
            /** SIMD width */
            static constexpr uint32_t simdSize = T_simdSize;

            /** maximum number of indices a worker must process if the domain is equally distributed over all worker */
            static constexpr uint32_t maxIndicesPerWorker
                = ((domainSize + simdSize * numWorkers - 1u) / (simdSize * numWorkers)) * simdSize;
        };
    } // namespace lockstep
} // namespace pmacc
