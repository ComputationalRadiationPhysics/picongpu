/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file definition of set temperature particle manipulator for forcing fixed electron temperature

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"
#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/manipulators/manipulators.hpp"

#include <pmacc/math/operation.hpp>

namespace picongpu::particles::atomicPhysics
{
    using SetTemperature = picongpu::particles::manipulators::unary::
        Temperature<picongpu::atomicPhysics::debug::scFlyComparison::TemperatureParam, pmacc::math::operation::Assign>;

} // namespace picongpu::particles::atomicPhysics
