/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace picongpu
{
    namespace traits
    {
        /** Get power of seven SI base units of date that is represented by an identifier
         *
         * Definition must follow the openPMD `unitDimension` definition:
         * length L, mass M, time T, electric current I, thermodynamic temperature
         * theta, amount of substance N, luminous intensity J
         *   @see http://www.openPMD.org
         *   @see http://dx.doi.org/10.5281/zenodo.33624
         * Must return a vector of size() == 7, for unitless attributes all
         * elements are zero.
         *
         * @tparam T_Identifier any picongpu identifier
         * @return \p std::vector<float_64> ::get() as static public method
         *
         */
        template<typename T_Identifier>
        struct UnitDimension;

    } /* namespace traits */

} /* namespace picongpu */
