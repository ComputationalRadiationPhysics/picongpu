/* Copyright 2019-2021 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Sergei Bastrakov, Klaus Steiniger
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

#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/cellType/Yee.hpp"
#include "picongpu/fields/incidentField/Solver.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/traits/GetStringProperties.hpp>

#include <memory>
#include <stdexcept>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Yee field solver with perfectly matched layer (PML) absorber
             *
             * Absorption is done using convolutional perfectly matched layer (CPML),
             * implemented according to [Taflove, Hagness].
             *
             * This class template is a public interface to be used, e.g. in .param
             * files and is compatible with other field solvers. Parameters of PML
             * are taken from pml.param, pml.unitless.
             *
             * Enabling this solver results in more memory being used on a device:
             * 12 additional scalar field values per each grid cell of a local domain.
             * Another limitation is not full persistency with checkpointing: the
             * additional values are not saved and so set to 0 after loading a
             * checkpoint (which in some cases still provides proper absorption, but
             * it is not guaranteed and results will differ due to checkpointing).
             *
             * This class template implements the general flow of CORE and BORDER field
             * updates and communication. The numerical schemes to perform the updates
             * are implemented by yeePML::detail::Solver.
             *
             * @tparam T_CurlE functor to compute curl of E
             * @tparam T_CurlB functor to compute curl of B
             */
            template<typename T_CurlE, typename T_CurlB>
            class YeePML
            {
            public:
                // Types required by field solver interface
                using CellType = cellType::Yee;
                using CurlE = T_CurlE;
                using CurlB = T_CurlB;

                YeePML(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                }

                /** Perform the first part of E and B propagation by a time step.
                 *
                 * Together with update_afterCurrent( ) forms the full propagation.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_beforeCurrent(uint32_t const currentStep)
                {
                }

                /** Perform the last part of E and B propagation by a time step
                 *
                 * Together with update_beforeCurrent( ) forms the full propagation.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_afterCurrent(uint32_t const currentStep)
                {
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "Yee");
                    return propList;
                }

            private:
                MappingDesc const cellDescription;
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for given field access in the YeePML solver
         *
         * It is always the same as the regular Yee solver with the given curl operators.
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         * @tparam T_Field field type
         */
        template<typename T_CurlE, typename T_CurlB, typename T_Field>
        struct GetMargin<fields::maxwellSolver::YeePML<T_CurlE, T_CurlB>, T_Field>
            : public GetMargin<fields::maxwellSolver::Yee<T_CurlE, T_CurlB>, T_Field>
        {
        };

    } // namespace traits

} // namespace picongpu
