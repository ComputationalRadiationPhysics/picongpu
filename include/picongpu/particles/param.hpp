/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
