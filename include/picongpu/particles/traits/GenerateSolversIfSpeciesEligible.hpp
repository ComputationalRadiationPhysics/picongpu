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

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <pmacc/meta/conversion/ToSeq.hpp>

#include <boost/mpl/apply.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/transform.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace traits
        {
            /** Return a list of Solvers specialized to all matching species
             *
             * Solvers can define the trait SpeciesEligibleForSolver to check a
             * particle species if it fulfills requirements of the solver.
             *
             * The compile-time factory here returns a list of particle solvers (of the
             * same solver given by T_Solver), but fully specialized with matching
             * particle species from a sequence of species (T_SeqSpecies).
             *
             * @tparam T_Solver a particle solver which shall be specialized for all
             *                  eligible particle species
             * @tparam T_SeqSpecies a sequence of particle species to check if they are
             *                      eligible to specialize T_Solver, also allows a
             *                      single type instead of a sequence
             * @tparam T_Eligible allows to specialize a solver but only if the check
             *                    of the T_Eligible class fulfills the
             *                    SpeciesEligibleForSolver trait, per default the
             *                    T_Solver argument is checked
             */
            template<typename T_Solver, typename T_SeqSpecies, typename T_Eligible = T_Solver>
            struct GenerateSolversIfSpeciesEligible
            {
                // wrap single arguments to sequence
                using SeqSpecies = typename pmacc::ToSeq<T_SeqSpecies>::type;
                // unspecialized solver
                using Solver = T_Solver;

                template<typename T_Species>
                struct Op : bmpl::apply1<Solver, T_Species>
                {
                };

                using SeqEligibleSpecies = typename bmpl::
                    copy_if<SeqSpecies, particles::traits::SpeciesEligibleForSolver<bmpl::_1, T_Eligible>>::type;

                using type = typename bmpl::transform<SeqEligibleSpecies, Op<bmpl::_1>>::type;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
