/**
 * Copyright 2016 Alexander Grund
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

    /** Uses the CUDA MRG32k3a RNG but does not store state members required for normal distribution*/
    class MRG32k3aMin
    {
    public:
        struct StateType
        {
            double s1[3];
            double s2[3];
        };

        DINLINE void
        init(StateType& state, uint32_t seed, uint32_t subsequence = 0, uint32_t offset = 0) const
        {
            curandStateMRG32k3a tmpState;
            curand_init(seed, subsequence, offset, &tmpState);
            AssignState(state, tmpState);
        }

        DINLINE uint32_t
        get32Bits(StateType& state) const
        {
            // We can do this cast if: 1) Only state data is used and
            //                         2) Data is aligned and positioned the same way
            return curand(reinterpret_cast<curandStateMRG32k3a*>(&state));
        }

        static std::string
        getName()
        {
            return "MRG32k3aMin";
        }
    private:
        // Sizes must match
        PMACC_STATIC_ASSERT_MSG(
                sizeof(StateType::s1) == sizeof(curandStateMRG32k3a::s1),
                Unexpected_sizes);
        PMACC_STATIC_ASSERT_MSG(
                sizeof(StateType::s2) == sizeof(curandStateMRG32k3a::s2),
                Unexpected_sizes);
        // Offsets must match
        PMACC_STATIC_ASSERT_MSG(
                offsetof(StateType, s1) == offsetof(curandStateMRG32k3a, s1) &&
                offsetof(StateType, s2) == offsetof(curandStateMRG32k3a, s2),
                Incompatible_structs);

        static HDINLINE void
        AssignState(StateType& dest, const curandStateMRG32k3a& src)
        {
            // Check if we can do this cast
            dest = reinterpret_cast<const StateType&>(src);
        }
    };

}  // namespace methods
}  // namespace random
}  // namespace PMacc
