/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/EMFieldBase.hpp"

#include <pmacc/algorithms/PromoteType.hpp>

#include <string>
#include <vector>

namespace picongpu
{
    /** Representation of the electric field
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldE : public fields::EMFieldBase
    {
    public:
        /** Create a field
         *
         * @param cellDescription mapping for kernels
         */
        FieldE(MappingDesc const& cellDescription);

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
