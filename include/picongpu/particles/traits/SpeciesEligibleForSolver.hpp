/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/meta/Mp11.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace traits
        {
            /** Check if species fulfills requirements of a solver
             *
             * Defines a pmacc::mp_bool true type is the particle species as all
             * requirements fulfilled for a solver.
             *
             * @tparam T_Species Species to check
             * @tparam T_Solver Solver with requirements
             */
            template<typename T_Species, typename T_Solver>
            struct SpeciesEligibleForSolver
            {
                using type = pmacc::mp_bool<true>;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
