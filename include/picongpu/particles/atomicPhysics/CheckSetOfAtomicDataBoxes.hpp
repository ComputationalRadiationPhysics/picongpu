/* Copyright 2024 Brian Marre
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

// need unit.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrderingFor.hpp"

#include <pmacc/static_assert.hpp>

namespace picongpu::particles::atomicPhysics
{
    struct CheckSetOfAtomicDataBoxes
    {
        template<
            typename T_TransitionOrdering,
            typename T_AtomicStateBoundFreeStartIndexBlockDataBox,
            typename T_AtomicStateBoundFreeNumberTransitionsDataBox,
            typename T_BoundFreeTransitionDataBox>
        static constexpr bool areBoundFreeAndOrdering()
        {
            PMACC_CASSERT_MSG(
                number_transition_dataBox_not_bound_free_based,
                u8(T_AtomicStateBoundFreeNumberTransitionsDataBox::processClassGroup)
                    == u8(enums::ProcessClassGroup::boundFreeBased));
            PMACC_CASSERT_MSG(
                startIndex_dataBox_not_bound_free_based,
                u8(T_AtomicStateBoundFreeStartIndexBlockDataBox::processClassGroup)
                    == u8(enums::ProcessClassGroup::boundFreeBased));
            PMACC_CASSERT_MSG(
                boundFreeTransitiondataBox_not_bound_free_based,
                u8(T_BoundFreeTransitionDataBox::processClassGroup) == u8(enums::ProcessClassGroup::boundFreeBased));
            PMACC_CASSERT_MSG(
                wrong_transition_ordering_boundFreeTransitionDataBox,
                u8(T_BoundFreeTransitionDataBox::transitionOrdering) == u8(T_TransitionOrdering));
            return true;
        }

        template<
            typename T_TransitionDirection,
            typename T_AtomicStateBoundFreeStartIndexBlockDataBox,
            typename T_AtomicStateBoundFreeNumberTransitionsDataBox,
            typename T_BoundFreeTransitionDataBox>
        static constexpr bool areBoundFreeAndDirection()
        {
            return areBoundFreeAndOrdering<
                s_enums::TransitionOrderingFor<T_TransitionDirection>::ordering,
                T_AtomicStateBoundFreeStartIndexBlockDataBox,
                T_AtomicStateBoundFreeNumberTransitionsDataBox,
                T_BoundFreeTransitionDataBox>();
        }
    };
} // namespace picongpu::particles::atomicPhysics
