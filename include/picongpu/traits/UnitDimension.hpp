/* Copyright 2015-2021 Axel Huebl
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

namespace picongpu
{
    namespace traits
    {
        /** Get power of seven SI base units of date that is represented by an identifier
         *
         * Definition must follow the openPMD `unitDimension` definition:
         * length L, mass M, time T, electric current I, thermodynamic temperature
         * theta, amount of substance N, luminous intensity J
         *   \see http://www.openPMD.org
         *   \see http://dx.doi.org/10.5281/zenodo.33624
         * Must return a vector of size() == 7, for unitless attributes all
         * elements are zero.
         *
         * \tparam T_Identifier any picongpu identifier
         * \return \p std::vector<float_64> ::get() as static public method
         *
         */
        template<typename T_Identifier>
        struct UnitDimension;

    } /* namespace traits */

} /* namespace picongpu */
