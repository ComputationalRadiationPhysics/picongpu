/* Copyright 2016-2019 Axel Huebl
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

#include "picongpu/traits/Unit.hpp"
#include "picongpu/traits/UnitDimension.hpp"

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

#include <string>
#include <vector>

namespace picongpu
{
namespace traits
{
    /** Reinterpret attributes for openPMD
     *
     * Currently, this conversion tables are used to translate the PIConGPU
     * totalCellIdx (unitless cell index) to the openPMD positionOffset (length)
     */
    template<typename T_Identifier>
    struct OpenPMDName;

    template<typename T_Identifier>
    struct OpenPMDUnit;

    template<typename T_Identifier>
    struct OpenPMDUnitDimension;

} // namespace traits
} // namespace picongpu

#include "PICToOpenPMD.tpp"
