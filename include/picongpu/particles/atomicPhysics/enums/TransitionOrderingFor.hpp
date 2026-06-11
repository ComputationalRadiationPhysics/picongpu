/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file get TransitionOrdering from TransitionDirection

#pragma once

#include "picongpu/particles/atomicPhysics/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"

namespace picongpu::particles::atomicPhysics::enums
{
    // error case, unknown is always false
    template<TransitionDirection T_TransitionDirection>
    struct TransitionOrderingFor;

    // upward case
    template<>
    struct TransitionOrderingFor<TransitionDirection::upward>
    {
        static constexpr TransitionOrdering ordering = TransitionOrdering::byLowerState;
    };

    // downward case
    template<>
    struct TransitionOrderingFor<TransitionDirection::downward>
    {
        static constexpr TransitionOrdering ordering = TransitionOrdering::byUpperState;
    };
} // namespace picongpu::particles::atomicPhysics::enums
