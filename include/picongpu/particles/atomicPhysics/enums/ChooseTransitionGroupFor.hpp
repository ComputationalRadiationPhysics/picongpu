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

//! @file get ChooseTransitionGroup from TransitionDirection and TransitionType

#pragma once

#include "picongpu/particles/atomicPhysics/enums/ChooseTransitionGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionType.hpp"

namespace picongpu::particles::atomicPhysics::enums
{
    // error case, unknown is always false
    template<TransitionType T_TransitionType, TransitionDirection T_TransitionDirection>
    struct ChooseTransitionGroupFor;

    //! bound-bound(upward)
    template<>
    struct ChooseTransitionGroupFor<TransitionType::boundBound, TransitionDirection::upward>
    {
        static constexpr ChooseTransitionGroup chooseTransitionGroup = ChooseTransitionGroup::boundBoundUpward;
    };

    //! bound-bound(downward)
    template<>
    struct ChooseTransitionGroupFor<TransitionType::boundBound, TransitionDirection::downward>
    {
        static constexpr ChooseTransitionGroup chooseTransitionGroup = ChooseTransitionGroup::boundBoundDownward;
    };

    //! bound-free(upward) may be either collisionalBoundFreeUpward or fieldBoundFreeUpward -> error to ask

    //! autonomous(downward)
    template<>
    struct ChooseTransitionGroupFor<TransitionType::autonomous, TransitionDirection::downward>
    {
        static constexpr ChooseTransitionGroup chooseTransitionGroup = ChooseTransitionGroup::autonomousDownward;
    };

    //! noChange
    //@{
    template<>
    struct ChooseTransitionGroupFor<TransitionType::noChange, TransitionDirection::upward>
    {
        static constexpr ChooseTransitionGroup chooseTransitionGroup = ChooseTransitionGroup::noChange;
    };
    template<>
    struct ChooseTransitionGroupFor<TransitionType::noChange, TransitionDirection::downward>
    {
        static constexpr ChooseTransitionGroup chooseTransitionGroup = ChooseTransitionGroup::noChange;
    };
    //@}
} // namespace picongpu::particles::atomicPhysics::enums
