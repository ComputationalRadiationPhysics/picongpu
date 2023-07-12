/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2
{
    //! check if TransitionType previously assigned by chooseTransitionType is valid
    template<typename T_Ion>
    HDINLINE void checkForInvalidTransitionType(T_Ion const ion)
    {
        if constexpr(picongpu::atomicPhysics2::debug::kernel::chooseTransition::CHECK_FOR_INVALID_TRANSITION_TYPE)
        {
            constexpr uint32_t maxValueTransitionTypeIndex
                = picongpu::particles::atomicPhysics2::enums::numberTransitionDataSets;

            if(!ion[accepted_] && (ion[transitionIndex_] >= maxValueTransitionTypeIndex))
                printf("atomicPhyiscs ERROR: detected invalid transitionType\n");
        }
    }
} // namespace picongpu::particles::atomicPhysics2
