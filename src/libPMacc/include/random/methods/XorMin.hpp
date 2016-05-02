/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc_types.hpp"
#include <curand_kernel.h>
#include <random/methods/Xor.hpp>

namespace PMacc
{
namespace random
{
namespace methods
{

    /** Uses the CUDA XORWOW RNG but does not store state members required for normal distribution*/
    class XorMin
    {
    public:
        class StateType
        {
        public:
            PMACC_ALIGN(d, unsigned int);
            PMACC_ALIGN(v[5], unsigned int);

            HDINLINE StateType()
            {}

            HDINLINE StateType(const curandStateXORWOW_t& other): d(other.d)
            {
                PMACC_STATIC_ASSERT_MSG(sizeof(v) == sizeof(other.v), Unexpected_sizes);
                for(unsigned i = 0; i < sizeof(v)/sizeof(v[0]); i++)
                    v[i] = other.v[i];
            }
        };

        DINLINE void
        init(StateType& state, uint32_t seed, uint32_t subsequence = 0, uint32_t offset = 0) const
        {
            curandStateXORWOW_t tmpState;
            curand_init(seed, subsequence, offset, &tmpState);
            state = tmpState;
        }

        HDINLINE uint32_t
        get32Bits(StateType& state) const
        {
            /* This generator uses the xorwow formula of
            www.jstatsoft.org/v08/i14/paper page 5
            Has period 2^192 - 2^32.
            */
            uint32_t t;
            t = (state.v[0] ^ (state.v[0] >> 2));
            state.v[0] = state.v[1];
            state.v[1] = state.v[2];
            state.v[2] = state.v[3];
            state.v[3] = state.v[4];
            state.v[4] = (state.v[4] ^ (state.v[4] <<4)) ^ (t ^ (t << 1));
            state.d += 362437;
            return state.v[4] + state.d;
        }

        static std::string
        getName()
        {
            return "XorMin";
        }
    };

}  // namespace methods
}  // namespace random
}  // namespace PMacc
