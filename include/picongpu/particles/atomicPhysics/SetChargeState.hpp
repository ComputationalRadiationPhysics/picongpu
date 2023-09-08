/* Copyright 2023-2023 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! functor for setting the charge state of a macro particle

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/SetToAtomicGroundState.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"

#include <pmacc/assert.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    struct SetChargeState
    {
        template<typename T_Ion>
        DINLINE void operator()(T_Ion& ion, float_X numberBoundElectrons)
        {
            PMACC_DEVICE_ASSERT_MSG(numberBoundElectrons >= 0._X, "Number of bound electrons must be >= 0");
            PMACC_DEVICE_ASSERT_MSG(
                numberBoundElectrons <= GetAtomicNumbers<T_Ion>::type::numberOfProtons,
                "Number of bound electrons must be <= numberOfProtons species");

            ion[boundElectrons_] = numberBoundElectrons;

            SetToAtomicGroundState{}(ion, static_cast<uint8_t>(numberBoundElectrons));
        }
    };
} // namespace picongpu::particles::atomicPhysics
