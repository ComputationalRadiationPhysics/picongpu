/* Copyright 2017-2021 Rene Widera
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
    namespace mappings
    {
        namespace threads
        {
            /** describe a constant index domain
             *
             * describe the size of the index domain and the number of workers to operate on the domain
             *
             * @tparam T_domainSize number of indices in the domain
             * @tparam T_workerSize number of worker working on @p T_domainSize
             * @tparam T_simdSize SIMD width
             */
            template<uint32_t T_domainSize, uint32_t T_workerSize, uint32_t T_simdSize = 1u>
            struct IdxConfig
            {
                /** number of indices within the domain */
                static constexpr uint32_t domainSize = T_domainSize;
                /** number of worker (threads) working on @p domainSize */
                static constexpr uint32_t workerSize = T_workerSize;
                /** SIMD width */
                static constexpr uint32_t simdSize = T_simdSize;

                /** number of collective iterations needed to address all indices */
                static constexpr uint32_t numCollIter
                    = (domainSize + simdSize * workerSize - 1u) / (simdSize * workerSize);
            };

        } // namespace threads
    } // namespace mappings
} // namespace pmacc
