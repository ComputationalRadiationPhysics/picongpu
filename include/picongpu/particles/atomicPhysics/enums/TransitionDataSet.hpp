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

/** @file transitionDataSet enum, enum of the internal storage groups of dataTransitions
 *
 * dataTransitions are internally organized by transitionType and transitionDirection(=^= TransitionOrdering,
 *  as defined in picongpu/particles/atomicPhyiscs2/enums/IsOrderRight.hpp).
 *
 * See @ref /picongpu/atomicPhysics/particles/atomicPhysics/TransitionType.hpp ["TransitionType.hpp"]
 *  for a definition of dataTransition.
 */

#pragma once

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::enums
{
    /** enum of used TransitionDataSets
     *
     * @attention noChange must always be second last entry!
     * @attention do not use custom values here, ChooseTransitionTypeKernel logic depends on continuous value
     * assignment starting at 0!
     */
    enum struct TransitionDataSet : uint32_t
    {
        boundBoundUpward, // = 0
        boundBoundDownward, // = 1
        boundFreeUpward, // = 2
        autonomousDownward, // = 3
        noChange, // = 4
        FINAL_NUMBER_ENTRIES // = 5
    };
    constexpr uint32_t numberTransitionDataSets = u32(TransitionDataSet::FINAL_NUMBER_ENTRIES);

} // namespace picongpu::particles::atomicPhysics::enums

namespace picongpu::particles::atomicPhysics
{
    template<enums::TransitionDataSet T_TransitionDataSet>
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(
            static_cast<uint8_t>(T_TransitionDataSet)
            == static_cast<uint8_t>(enums::TransitionDataSet::boundBoundUpward))
            return "bound-bound(upward)";
        if constexpr(
            static_cast<uint8_t>(T_TransitionDataSet)
            == static_cast<uint8_t>(enums::TransitionDataSet::boundBoundDownward))
            return "bound-bound(downward)";
        if constexpr(
            static_cast<uint8_t>(T_TransitionDataSet)
            == static_cast<uint8_t>(enums::TransitionDataSet::boundFreeUpward))
            return "bound-free(upward)";
        if constexpr(
            static_cast<uint8_t>(T_TransitionDataSet)
            == static_cast<uint8_t>(enums::TransitionDataSet::autonomousDownward))
            return "autonomous(downward)";
        if constexpr(
            static_cast<uint8_t>(T_TransitionDataSet) == static_cast<uint8_t>(enums::TransitionDataSet::noChange))
            return "noChange";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
