/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include "picongpu/fields/FieldB.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/fields/EMFieldBase.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <string>
#include <type_traits>
#include <vector>


namespace picongpu
{
    FieldB::FieldB(MappingDesc const& cellDescription)
        : fields::EMFieldBase(
            cellDescription,
            getName(),
            traits::GetLowerMargin<fields::Solver, FieldB>::type::toRT(),
            traits::GetUpperMargin<fields::Solver, FieldB>::type::toRT())
    {
    }

    FieldB::UnitValueType FieldB::getUnit()
    {
        return UnitValueType{sim.unit.bField(), sim.unit.bField(), sim.unit.bField()};
    }

    std::vector<float_64> FieldB::getUnitDimension()
    {
        /* B is in Tesla : kg / (A * s^2)
         *   -> M * T^-2 * I^-1
         */
        std::vector<float_64> unitDimension(7, 0.0);
        unitDimension.at(SIBaseUnits::mass) = 1.0;
        unitDimension.at(SIBaseUnits::time) = -2.0;
        unitDimension.at(SIBaseUnits::electricCurrent) = -1.0;
        return unitDimension;
    }

    std::string FieldB::getName()
    {
        return "B";
    }

} // namespace picongpu
