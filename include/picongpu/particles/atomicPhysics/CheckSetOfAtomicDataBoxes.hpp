/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Brian Marre
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
#include "picongpu/particles/atomicPhysics/enums/IsProcess.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrderingFor.hpp"

#include <pmacc/static_assert.hpp>

namespace picongpu::particles::atomicPhysics
{
    namespace s_enums = picongpu::particles::atomicPhysics::enums;

    struct CheckSetOfAtomicDataBoxes
    {
        template<
            enums::TransitionOrdering T_TransitionOrdering,
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
            enums::TransitionDirection T_TransitionDirection,
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

        //! check that the processClass matches the TransitionDataBox passed in TransitionOrdering and TransitionType
        template<s_enums::ProcessClass T_ProcessClass, typename T_TransitionDataBox>
        static constexpr bool transitionDataBoxMatchesProcessClass()
        {
            PMACC_CASSERT_MSG(
                transition_dataBox_and_processClass_inconsistent,
                s_enums::IsProcess<T_TransitionDataBox::processClassGroup>::check(u8(T_ProcessClass)));

            constexpr bool isUpward
                = s_enums::IsProcess<s_enums::ProcessClassGroup::upward>::check(u8(T_ProcessClass));

            /* check ordering consistent with direction of T_ProcessClass, otherwise unphysical behaviour and/or
             * illegal memory accesses */
            if constexpr(isUpward)
            {
                PMACC_CASSERT_MSG(
                    transition_databOxOrdering_and_processClass_inconsistent,
                    T_TransitionDataBox::transitionOrdering == s_enums::TransitionOrdering::byLowerState);
            }
            else
            {
                PMACC_CASSERT_MSG(
                    transition_databOxOrdering_and_processClass_inconsistent,
                    T_TransitionDataBox::transitionOrdering == s_enums::TransitionOrdering::byUpperState);
            }

            return true;
        }
    };
} // namespace picongpu::particles::atomicPhysics
