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

//! @file definition of set temperature particle manipulator for forcing fixed electron temperature

#pragma once

#include "picongpu/defines.hpp"
// need atomicPhysics_Debug.param

#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/manipulators/manipulators.hpp"

#include <pmacc/math/operation.hpp>

namespace picongpu::particles::atomicPhysics
{
    using SetTemperature = picongpu::particles::manipulators::unary::
        Temperature<picongpu::atomicPhysics::debug::scFlyComparison::TemperatureParam, pmacc::math::operation::Assign>;

} // namespace picongpu::particles::atomicPhysics
