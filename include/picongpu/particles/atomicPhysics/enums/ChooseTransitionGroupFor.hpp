/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file get ChooseTransitionGroup token from TransitionDirection token and TransitionType token
 *
 * File contains one specialisation of ChooseTransitionGroupFor for every combination of TransitionType and
 *  TransitionDirection, giving access to the corresponding ChooseTransitionGroup,
 *
 * @attention no ChooseTransitionGroupFor specialization exists for **bound-free(upward)**, since the
 *  ChooseTransitionGroup for bound-free(upward) may be either collisionalBoundFreeUpward or fieldBoundFreeUpward and
 *  therefore no single ChooseTransitionGroup may be defined.
 */

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
