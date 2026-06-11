/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file chooseTransitionGroup, enum of groups of transitions used by the ChooseTransitionGroupKernel.
 *
 * Macro ions are subdivided into these groups in the ChooseTransitionGroupKernel and later assigned a specific
 * transition in the ChooseTransitionKernel from the group of transition they have been assigned.
 *
 * @attention The ChooseTransition grouping differs from the transition storage grouping in the atomicData object.
 *  Transition data is organized by transitionType and transitionDirection, see
 *  @ref /picongpu/atomicPhysics/particles/atomicPhysics/TransitionType.hpp ["TransitionType.hpp"] for the grouping
 *  of transition data.
 */

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::enums
{
    /** enum of used ChooseTransitionGroups
     *
     * for every entry before FINAL_NUMBER_DATE_CACHE_DATA_SETS the rate cache will have one per state rate data set,
     *  while everything after excluding the second last entry will, get one entry for all states.
     *
     * @attention noChange must always be second last entry!
     * @attention do not use custom values here, ChooseTransitionGroupKernel logic depends on continuous value
     * assignment starting at 0!
     */
    enum struct ChooseTransitionGroup : uint32_t
    {
        boundBoundUpward, // = 0
        boundBoundDownward, // = 1
        collisionalBoundFreeUpward, // = 2
        autonomousDownward, // = 3
        fieldBoundFreeUpward, // = 4
        noChange, // = 5
        FINAL_NUMBER_ENTRIES // = 6
    };
    constexpr uint32_t numberChooseTransitionGroups = u32(ChooseTransitionGroup::FINAL_NUMBER_ENTRIES);
} // namespace picongpu::particles::atomicPhysics::enums

namespace picongpu::particles::atomicPhysics
{
    template<enums::ChooseTransitionGroup T_ChooseTransitionGroup>
    std::string enumToString()
    {
        if constexpr(u32(T_ChooseTransitionGroup) == u32(enums::ChooseTransitionGroup::boundBoundUpward))
            return "bound-bound(upward)";
        if constexpr(u32(T_ChooseTransitionGroup) == u32(enums::ChooseTransitionGroup::boundBoundDownward))
            return "bound-bound(downward)";
        if constexpr(u32(T_ChooseTransitionGroup) == u32(enums::ChooseTransitionGroup::collisionalBoundFreeUpward))
            return "collisional bound-free(upward)";
        if constexpr(u32(T_ChooseTransitionGroup) == u32(enums::ChooseTransitionGroup::fieldBoundFreeUpward))
            return "field bound-free(upward)";
        if constexpr(u32(T_ChooseTransitionGroup) == u32(enums::ChooseTransitionGroup::autonomousDownward))
            return "autonomous(downward)";
        if constexpr(u32(T_ChooseTransitionGroup) == u32(enums::ChooseTransitionGroup::noChange))
            return "noChange";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
