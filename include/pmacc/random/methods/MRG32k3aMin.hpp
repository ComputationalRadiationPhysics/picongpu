/* Copyright 2016-2021 Alexander Grund, Rene Widera
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
#include "pmacc/static_assert.hpp"

#if(PMACC_CUDA_ENABLED != 1)
#    include "pmacc/random/methods/AlpakaRand.hpp"
#else
#    include <curand_kernel.h>
#endif


namespace pmacc
{
    namespace random
    {
        namespace methods
        {
#if(PMACC_CUDA_ENABLED != 1)
            //! fallback to alpaka RNG if a cpu accelerator is used
            template<typename T_Acc = cupla::Acc>
            using MRG32k3aMin = AlpakaRand<T_Acc>;
#else
            //! Mersenne-Twister random number generator with a reduced state
            template<typename T_Acc = cupla::Acc>
            class MRG32k3aMin
            {
            public:
                struct StateType
                {
                    double s1[3];
                    double s2[3];
                };

                DINLINE void init(T_Acc const& acc, StateType& state, uint32_t seed, uint32_t subsequence = 0) const
                {
                    curandStateMRG32k3a tmpState;
                    curand_init(seed, subsequence, 0, &tmpState);
                    AssignState(state, tmpState);
                }

                DINLINE uint32_t get32Bits(T_Acc const& acc, StateType& state) const
                {
                    /* We can do this cast if: 1) Only state data is used and
                     *                         2) Data is aligned and positioned the same way
                     */
                    return curand(reinterpret_cast<curandStateMRG32k3a*>(&state));
                }

                DINLINE uint64_t get64Bits(T_Acc const& acc, StateType& state) const
                {
                    // two 32bit values are packed into a 64bit value
                    uint64_t result = get32Bits(acc, state);
                    result <<= 32;
                    result ^= get32Bits(acc, state);
                    return result;
                }

                static std::string getName()
                {
                    return "MRG32k3aMin";
                }

            private:
                // Sizes must match
                PMACC_STATIC_ASSERT_MSG(sizeof(StateType::s1) == sizeof(curandStateMRG32k3a::s1), Unexpected_sizes);
                PMACC_STATIC_ASSERT_MSG(sizeof(StateType::s2) == sizeof(curandStateMRG32k3a::s2), Unexpected_sizes);
                // Offsets must match
                PMACC_STATIC_ASSERT_MSG(
                    offsetof(StateType, s1) == offsetof(curandStateMRG32k3a, s1)
                        && offsetof(StateType, s2) == offsetof(curandStateMRG32k3a, s2),
                    Incompatible_structs);

                static DINLINE void AssignState(StateType& dest, curandStateMRG32k3a const& src)
                {
                    // Check if we can do this cast
                    dest = reinterpret_cast<StateType const&>(src);
                }
            };
#endif
        } // namespace methods
    } // namespace random
} // namespace pmacc
