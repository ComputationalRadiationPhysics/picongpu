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

//! @file get TransitionDataSet from TransitionDirection and TransitionType

#pragma once

#include "picongpu/particles/atomicPhysics2/enums/TransitionDataSet.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionType.hpp"

namespace picongpu::particles::atomicPhysics2::enums
{
    // error case, unknown is always false
    template<TransitionType T_TransitionType, TransitionDirection T_TransitionDirection>
    struct TransitionDataSetFor;

    //! bound-bound(upward)
    template<>
    struct TransitionDataSetFor<TransitionType::boundBound, TransitionDirection::upward>
    {
        static constexpr TransitionDataSet dataSet = TransitionDataSet::boundBoundUpward;
    };

    //! bound-bound(downward)
    template<>
    struct TransitionDataSetFor<TransitionType::boundBound, TransitionDirection::downward>
    {
        static constexpr TransitionDataSet dataSet = TransitionDataSet::boundBoundDownward;
    };

    //! bound-free(upward)
    template<>
    struct TransitionDataSetFor<TransitionType::boundFree, TransitionDirection::upward>
    {
        static constexpr TransitionDataSet dataSet = TransitionDataSet::boundFreeUpward;
    };

    //! autonomous(downward)
    template<>
    struct TransitionDataSetFor<TransitionType::autonomous, TransitionDirection::downward>
    {
        static constexpr TransitionDataSet dataSet = TransitionDataSet::autonomousDownward;
    };

    //! noChange
    //@{
    template<>
    struct TransitionDataSetFor<TransitionType::noChange, TransitionDirection::upward>
    {
        static constexpr TransitionDataSet dataSet = TransitionDataSet::noChange;
    };
    template<>
    struct TransitionDataSetFor<TransitionType::noChange, TransitionDirection::downward>
    {
        static constexpr TransitionDataSet dataSet = TransitionDataSet::noChange;
    };
    //@}
} // namespace picongpu::particles::atomicPhysics2::enums
