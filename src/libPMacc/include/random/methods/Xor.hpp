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

namespace PMacc
{
namespace random
{
namespace methods
{

    /** Uses the CUDA XORWOW RNG */
    class Xor
    {
    public:
        typedef curandStateXORWOW_t StateType;

        DINLINE void
        init(StateType& state, uint32_t seed, uint32_t subsequence = 0, uint32_t offset = 0) const
        {
            curand_init(seed, subsequence, offset, &state);
        }

        DINLINE uint32_t
        get32Bits(StateType& state) const
        {
            return curand(&state);
        }

        static std::string
        getName()
        {
            return "Xor";
        }
    };

}  // namespace methods
}  // namespace random
}  // namespace PMacc
