/**
 * Copyright 2016-2017 Alexander Grund
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

    /** Uses the CUDA MRG32k3a RNG */
    class MRG32k3a
    {
    public:
        typedef curandStateMRG32k3a StateType;

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
            return "MRG32k3a";
        }
    };

}  // namespace methods
}  // namespace random
}  // namespace PMacc
