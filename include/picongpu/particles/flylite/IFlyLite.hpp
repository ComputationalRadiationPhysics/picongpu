/* Copyright 2017-2021 Axel Huebl
 *
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

// pmacc
#include <pmacc/static_assert.hpp>

#include <boost/core/ignore_unused.hpp>
#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            /** Interface for a method of solving population kinetics
             */
            class IFlyLite
            {
            public:
                /** Allocate & Initialize Memory Buffers for Algorithms
                 *
                 * @param gridSizeLocal local size of electro-magnetic fields on the cells
                 * @param ionSpeciesName unique name for the ion species
                 */
                virtual void init(pmacc::DataSpace<simDim> const& gridSizeLocal, std::string const& ionSpeciesName)
                    = 0;

                /** Calculate Evolution of Populations for One Time Step
                 *
                 * Interface for the update of the atomic populations during the PIC
                 * cycle.
                 *
                 * @param ionSpeciesName unique name for the ion species
                 * @param currentStep the current time step of the simulation
                 */
                template<typename T_IonSpecies>
                void update(std::string const& ionSpeciesName, uint32_t currentStep)
                {
                    boost::ignore_unused(ionSpeciesName, currentStep);
                    /* The compiler is allowed to evaluate an expression those not depends on a template parameter
                     * even if the class is never instantiated. In that case static assert is always
                     * evaluated (e.g. with clang), this results in an error if the condition is false.
                     * http://www.boost.org/doc/libs/1_60_0/doc/html/boost_staticassert.html
                     *
                     * A workaround is to add a template dependency to the expression.
                     * `sizeof(ANY_TYPE) != 0` is always true and defers the evaluation.
                     */
                    PMACC_STATIC_ASSERT_MSG(
                        false && sizeof(T_IonSpecies) != 0,
                        FLYlite_the_update_method_for_ion_population_kinetics_is_not_implemented, );
                }
            };
        } // namespace flylite
    } // namespace particles
} // namespace picongpu
