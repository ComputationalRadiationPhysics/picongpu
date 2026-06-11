/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt, Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "picongpu/fields/FieldE.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/fields/EMFieldBase.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/simulation_types.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <string>
#include <type_traits>
#include <vector>

namespace picongpu
{
    FieldE::FieldE(MappingDesc const& cellDescription)
        : fields::EMFieldBase(
              cellDescription,
              getName(),
              picongpu::traits::GetLowerMargin<fields::Solver, FieldE>::type::toRT(),
              picongpu::traits::GetUpperMargin<fields::Solver, FieldE>::type::toRT())
    {
    }

    FieldE::UnitValueType FieldE::getUnit()
    {
        return UnitValueType{sim.unit.eField(), sim.unit.eField(), sim.unit.eField()};
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
