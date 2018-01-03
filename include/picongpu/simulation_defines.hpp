/* Copyright 2013-2018 Rene Widera
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


#pragma once

#include <stdint.h>
#include <pmacc/types.hpp>
#include <picongpu/simulation_types.hpp>
#include "pmacc_renamings.hpp"


namespace picongpu
{
    using namespace pmacc;
}

/* IMPORTANT we need to use #include <...> for local files
 * else we get problems with our EXTENTION_PATH from cmake which
 * overwrites local defined include files.
 */

//##### load param
#include <picongpu/simulation_defines/_defaultParam.loader>
#include <picongpu/simulation_defines/extensionParam.loader>

//load starter after all user extension
#include <picongpu/simulation_defines/param/starter.param>

#include <picongpu/simulation_defines/param/components.param>
#include <picongpu/simulation_classTypes.hpp>

// ##### load unitless
#include <picongpu/simulation_defines/_defaultUnitless.loader>
#include <picongpu/simulation_defines/extensionUnitless.loader>
//load starter after user extensions and all params are loaded
#include <picongpu/simulation_defines/unitless/starter.unitless>
