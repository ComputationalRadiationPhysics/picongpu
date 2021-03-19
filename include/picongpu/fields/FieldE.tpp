/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
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

#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/EMFieldBase.hpp"
#include "picongpu/simulation_types.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <string>
#include <vector>
#include <type_traits>


namespace picongpu
{
    FieldE::FieldE(MappingDesc const& cellDescription) : fields::EMFieldBase<FieldE>(cellDescription, getName())
    {
    }

    HDINLINE FieldE::UnitValueType FieldE::getUnit()
    {
        return UnitValueType{UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD};
    }

    std::vector<float_64> FieldE::getUnitDimension()
    {
        /* E is in volts per meters: V / m = kg * m / (A * s^3)
         *   -> L * M * T^-3 * I^-1
         */
        std::vector<float_64> unitDimension(7, 0.0);
        unitDimension.at(SIBaseUnits::length) = 1.0;
        unitDimension.at(SIBaseUnits::mass) = 1.0;
        unitDimension.at(SIBaseUnits::time) = -3.0;
        unitDimension.at(SIBaseUnits::electricCurrent) = -1.0;
        return unitDimension;
    }

    std::string FieldE::getName()
    {
        return "E";
    }

} // namespace picongpu
