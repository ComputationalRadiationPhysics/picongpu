/* Copyright 2013-2023 Rene Widera
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

// clang-format off

#include <cstdint>
#include <pmacc/types.hpp>
#include <picongpu/simulation_types.hpp>
#include "pmacc_renamings.hpp"
#include "picongpu/traits/GetMargin.hpp"


namespace picongpu
{
    using namespace pmacc;
    using namespace picongpu::traits;
}

/* IMPORTANT we need to use #include <...> for local files
 * else we get problems with our EXTENTION_PATH from cmake which
 * overwrites local defined include files.
 */

//##### load param
#include <picongpu/_defaultParam.loader>
#include <picongpu/extensionParam.loader>

#include <picongpu/simulation_classTypes.hpp>

// ##### load unitless
#include <picongpu/_defaultUnitless.loader>
#include <picongpu/extensionUnitless.loader>

#include "picongpu/fields/Fields.tpp"
#include <pmacc/particles/IdProvider.hpp>

// clang-format on
