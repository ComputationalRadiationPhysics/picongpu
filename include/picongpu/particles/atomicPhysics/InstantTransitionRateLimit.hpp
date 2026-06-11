/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
