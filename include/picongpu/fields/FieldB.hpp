/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Sergei Bastrakov
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
#include "picongpu/fields/EMFieldBase.hpp"

#include <pmacc/algorithms/PromoteType.hpp>

#include <string>
#include <vector>

namespace picongpu
{
    /** Representation of the magnetic field
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldB : public fields::EMFieldBase
    {
    public:
        /** Create a field
         *
         * @param cellDescription mapping for kernels
         */
        FieldB(MappingDesc const& cellDescription);

        //! Unit type of field components
        using UnitValueType = promoteType<float_64, ValueType>::type;

        //! Get units of field components
        static UnitValueType getUnit();

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */
        static std::vector<float_64> getUnitDimension();

        //! Get text name
        static std::string getName();
    };

} // namespace picongpu
