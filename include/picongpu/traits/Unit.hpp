/* Copyright 2013-2021 Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"

namespace picongpu
{
    namespace traits
    {
        /** Get unit of a date that is represented by an identifier
         *
         * @tparam T_Identifier any PIConGPU identifier
         * @return \p std::vector<float_64> ::get() as static public method
         *
         * Unitless identifies, see \UnitDimension, can still be scaled by a
         * factor. If they are not scaled, implement the unit as 1.0;
         * @see unitless/speciesAttributes.unitless
         */
        template<typename T_Identifier, typename T_Parameter = void>
        struct Unit
        {
            static auto get()
            {
                return T_Identifier::getUnit();
            }
        };

        /// TODO: make this nicer. It works in principle, but is not pretty now.
        /// Fields need to have a unit accessible on host and device as a pmacc vector.
        /// So for those types we hack in the unit thing accordingly.

        //! Host-device unit accessor for FieldE
        template<>
        struct Unit<FieldE>
        {
            HDINLINE static auto get()
            {
                return FieldE::UnitValueType{UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD};
            }
        };

        //! Host-device unit accessor for FieldB
        template<>
        struct Unit<FieldB>
        {
            HDINLINE static auto get()
            {
                return FieldB::UnitValueType{UNIT_BFIELD, UNIT_BFIELD, UNIT_BFIELD};
            }
        };

        //! Host-device unit accessor for FieldJ
        template<>
        struct Unit<FieldJ>
        {
            HDINLINE static auto get()
            {
                const float_64 UNIT_CURRENT = UNIT_CHARGE / UNIT_TIME / (UNIT_LENGTH * UNIT_LENGTH);
                return FieldJ::UnitValueType{UNIT_CURRENT, UNIT_CURRENT, UNIT_CURRENT};
            }
        };

        //! Host-device unit accessor for FieldTmp and the given frame solver type
        template<typename T_FrameSolver>
        struct Unit<FieldTmp, T_FrameSolver>
        {
            HDINLINE static auto get()
            {
                return T_FrameSolver{}.getUnit();
            }
        };

    } // namespace traits

} // namespace picongpu
