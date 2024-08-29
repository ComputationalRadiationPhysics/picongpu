/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    using namespace pmacc;
}

// clang-format off
#include "picongpu/param/precision.param"
#include "picongpu/param/dimension.param"
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include "picongpu/param/mallocMC.param"
#endif
#include "picongpu/param/memory.param"
#include "picongpu/param/random.param"
#include "picongpu/param/physicalConstants.param"
#include "picongpu/param/speciesConstants.param"
#include "picongpu/param/simulation.param"
#include "picongpu/param/unit.param"
#include "picongpu/unitless/simulation.unitless"
#include "picongpu/unitless/physicalConstants.unitless"
#include "picongpu/unitless/speciesConstants.unitless"
// clang-format on
