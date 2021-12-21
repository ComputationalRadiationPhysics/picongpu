/* Copyright 2017-2021 Axel Huebl
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

#include <boost/mpl/bool.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace traits
        {
            /** Check if species fulfills requirements of a solver
             *
             * Defines a boost::mpl::bool_ true type is the particle species as all
             * requirements fulfilled for a solver.
             *
             * @tparam T_Species Species to check
             * @tparam T_Solver Solver with requirements
             */
            template<typename T_Species, typename T_Solver>
            struct SpeciesEligibleForSolver
            {
                using type = boost::mpl::bool_<true>;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
