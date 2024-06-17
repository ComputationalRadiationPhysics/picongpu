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
