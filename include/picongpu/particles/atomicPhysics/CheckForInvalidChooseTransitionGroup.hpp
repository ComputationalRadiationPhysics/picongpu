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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"
#include "picongpu/particles/atomicPhysics/enums/ChooseTransitionGroup.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    //! check if ChooseTransitionGroup previously assigned by ChooseTransitionGroupKernel is valid
    template<typename T_Ion>
    HDINLINE void checkForInvalidChooseTransitionGroup(T_Ion const ion)
    {
        if constexpr(picongpu::atomicPhysics::debug::kernel::chooseTransition::CHECK_FOR_INVALID_TRANSITION_TYPE)
        {
            constexpr uint32_t maxValueChooseTransitionGroupIndex
                = picongpu::particles::atomicPhysics::enums::numberChooseTransitionGroups;

            if(!ion[accepted_] && (ion[transitionIndex_] >= maxValueChooseTransitionGroupIndex))
                printf("atomicPhyiscs ERROR: detected invalid chooseTransitionGroup\n");
        }
    }
} // namespace picongpu::particles::atomicPhysics
