/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
    HDINLINE void checkForInvalidChooseTransitionGroup([[maybe_unused]] T_Ion const ion)
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
