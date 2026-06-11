/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file compilation of all local helper fields for the calculation ionization potential depression(IPD)
 *
 * hold quantity for each super cell
 *
 * @details used for calculating IPD
 */


#pragma once

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/DebyeLengthField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/FreeElectronDensityField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/TemperatureEnergyField.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/localHelperFields/ZStarField.hpp"
