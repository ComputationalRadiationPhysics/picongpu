/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt, Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
