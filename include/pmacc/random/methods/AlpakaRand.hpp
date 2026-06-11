/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        namespace methods
        {
            template<typename T_Acc = pmacc::Acc<DIM1>>
            class AlpakaRand
            {
            public:
                using StateType = decltype(::alpaka::rand::engine::createDefault(
                    alpaka::core::declval<T_Acc const&>(),
                    alpaka::core::declval<uint32_t&>(),
                    alpaka::core::declval<uint32_t&>()));

                template<typename T_Worker>
                DINLINE void init(T_Worker const& worker, StateType& state, uint32_t seed, uint32_t subsequence = 0)
                    const
                {
                    state = ::alpaka::rand::engine::createDefault(worker.getAcc(), seed, subsequence);
                }

                template<typename T_Worker>
                DINLINE uint32_t get32Bits(T_Worker const& worker, StateType& state) const
                {
                    return ::alpaka::rand::distribution::createUniformUint<uint32_t>(worker.getAcc())(state);
                }

                template<typename T_Worker>
                DINLINE uint64_t get64Bits(T_Worker const& worker, StateType& state) const
                {
                    /* Two 32bit values are packed into a 64bit value because alpaka is not
                     * supporting 64bit integer random numbers
                     */
                    uint64_t result = get32Bits(worker, state);
                    result <<= 32;
                    result ^= get32Bits(worker, state);
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
