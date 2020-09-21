/* Copyright 2020 Marco Garten, Brian Marre
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

#include <picongpu/traits/UsesRNG.hpp> // (1)

namespace picongpu
{
namespace traits
{
    /** specialization of the UsesRNG trait for atomicPhysics rate equations
     * solvers
     *
     * UsesRNG defined in (1)
     *
     * This trait is used in MySimulation to check whether a random number
     * generator(RNG) has to be created, since RNGs need memory
     *
     * since atomic Phyisics Modules can use random number generation a
     * specialization is required for them
     */
    template<
        typename T_RateEquationAlgorithm,
        typename T_IonSpecies,
        typename T_ElectronSpecies
        >
    struct UsesRNG< particles::atomicPhysics::rateEquationSolver::RateEquationSolver<
            T_RateEquationAlgorithm,
            T_IonSpecies,
            T_ElectronSpecies
            >
        > :
    public boost::true_type
    {
    };
    
} // namespace traits
} // namespace picongpu
