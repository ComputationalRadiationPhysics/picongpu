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

/** @file compilation of all sum helper fields for ionization potential depression(IPD)
 *
 * hold sum of quantity for each super cell
 *
 * @details used for accumulating quantities for calcuation of IPD input
 */


#pragma once

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/SumChargeNumberIonsField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/SumChargeNumberSquaredIonsField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/SumTemperatureFunctionalField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/SumWeightAllField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/SumWeightElectronsField.hpp"
