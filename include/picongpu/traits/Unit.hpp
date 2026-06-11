/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

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
        template<typename T_Identifier>
        struct Unit;

    } // namespace traits

} // namespace picongpu
