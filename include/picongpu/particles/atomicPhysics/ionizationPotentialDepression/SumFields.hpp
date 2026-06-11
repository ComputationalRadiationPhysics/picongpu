/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
