/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

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

// need unit system and normalization
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

namespace picongpu::particles::atomicPhysics
{
    struct InstantTransitionRateLimit
    {
        /** get maximum of total state loss rate for inclusion in the time dependent rate equation solver
         *
         * @tparam T_ReturnType type and precision to use in the result
         */
        template<typename T_ReturnType>
        static constexpr T_ReturnType get()
        {
            using picongpu::atomicPhysics::RateSolverParam;

            // unit: unitless * unitless / unit_time = 1/unit_time
            return static_cast<T_ReturnType>(
                       RateSolverParam::timeStepAlpha * float_X(RateSolverParam::maximumNumberSubStepsPerPICTimeStep))
                   / picongpu::sim.pic.getDt<T_ReturnType>();
        }
    };
} // namespace picongpu::particles::atomicPhysics
