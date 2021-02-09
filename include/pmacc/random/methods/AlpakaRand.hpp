/* Copyright 2015-2021 Alexander Grund, Rene Widera
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
    namespace random
    {
        namespace methods
        {
            template<typename T_Acc = cupla::Acc>
            class AlpakaRand
            {
            public:
                using StateType = decltype(::alpaka::rand::generator::createDefault(
                    alpaka::core::declval<T_Acc const&>(),
                    alpaka::core::declval<uint32_t&>(),
                    alpaka::core::declval<uint32_t&>()));

                DINLINE void init(T_Acc const& acc, StateType& state, uint32_t seed, uint32_t subsequence = 0) const
                {
                    state = ::alpaka::rand::generator::createDefault(acc, seed, subsequence);
                }

                DINLINE uint32_t get32Bits(T_Acc const& acc, StateType& state) const
                {
                    return ::alpaka::rand::distribution::createUniformUint<uint32_t>(acc)(state);
                }

                DINLINE uint64_t get64Bits(T_Acc const& acc, StateType& state) const
                {
                    /* Two 32bit values are packed into a 64bit value because alpaka is not
                     * supporting 64bit integer random numbers
                     */
                    uint64_t result = get32Bits(acc, state);
                    result <<= 32;
                    result ^= get32Bits(acc, state);
                    return result;
                }

                static std::string getName()
                {
                    return "AlpakaRand";
                }
            };

        } // namespace methods
    } // namespace random
} // namespace pmacc
