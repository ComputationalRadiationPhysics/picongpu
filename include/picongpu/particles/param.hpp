/* Copyright 2024 Rene Widera
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/pusher/param.hpp"

// clang-format off
#include "picongpu/particles/ionization/param.hpp"
#include "picongpu/param/density.param"
#include "picongpu/param/particle.param"
#include "picongpu/particles/atomicPhysics/param.hpp"
#include "picongpu/param/particleFilters.param"
#include "picongpu/param/species.param"
#include "picongpu/param/speciesDefinition.param"
#include "picongpu/param/collision.param"
#include "picongpu/param/fieldSolver.param"

#include "picongpu/unitless/density.unitless"
#include "picongpu/unitless/particle.unitless"
#include "picongpu/unitless/ionizer.unitless"
#include "picongpu/unitless/speciesAttributes.unitless"
#include "picongpu/unitless/speciesDefinition.unitless"
#include "picongpu/unitless/collision.unitless"
// clang-format on

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/ApplyIPDIonization.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Electron.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Ion.hpp"