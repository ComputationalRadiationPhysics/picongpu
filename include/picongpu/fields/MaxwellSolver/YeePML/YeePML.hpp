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
#include "picongpu/fields/absorber/pml/Solver.hpp"
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

                YeePML(MappingDesc const cellDescription) : cellDescription(cellDescription), solver(cellDescription)
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
                    /* These steps are the same as in the Yee solver, PML updates are done as part of methods of
                     * solver. Note that here we do the second half of updating B, thus completing the first half
                     * started in a call to update_afterCurrent() at the previous time step. This splitting of B update
                     * is standard for Yee-type field solvers in PIC codes due to particle pushers normally requiring E
                     * and B values defined at the same time while the field solver operates with time-staggered
                     * fields. However, while the standard Yee solver in vacuum is linear in a way of two consecutive
                     * updates by dt/2 being equal to one update by dt, this is not true for the convolutional field
                     * updates in PML. Thus, for PML we have to distinguish between the updates by dt/2 by introducing
                     * first and second halves of the update. This distinction only concerns the convolutional field B
                     * data used inside the PML, and not the full fields used by the rest of the code. In the very
                     * first time step of a simulation we start with the second half right away, but this is no
                     * problem, since the only meaningful initial conditions in the PML area are zero for the
                     * to-be-absorbed components.
                     */
                    solver.template updateBSecondHalf<CORE + BORDER>(currentStep);
                    auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                    // update B by half step, to step = currentStep + 0.5, so step for E_inc = currentStep
                    incidentFieldSolver.updateBHalf(static_cast<float_X>(currentStep));
                    auto& fieldB = solver.getFieldB();
                    EventTask eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());

                    solver.template updateE<CORE>(currentStep);
                    // Incident field solver update does not use exchanged B, so does not have to wait for it
                    // update E by full step, to step = currentStep + 1, so step for B_inc = currentStep + 0.5
                    incidentFieldSolver.updateE(static_cast<float_X>(currentStep) + 0.5_X);
                    __setTransactionEvent(eRfieldB);
                    solver.template updateE<BORDER>(currentStep);
                }

                /** Perform the last part of E and B propagation by a time step
                 *
                 * Together with update_beforeCurrent( ) forms the full propagation.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_afterCurrent(uint32_t const currentStep)
                {
                    /* These steps are the same as in the Yee solver, except the Fabsorber::ExponentialDamping::run( )
                     * is not called, PML updates are done as part of calls to methods of solver. As explained in more
                     * detail in comments inside update_beforeCurrent(), here we start a new step of updating B in
                     * terms of the time-staggered Yee grid. And so this is the first half of B update, to be completed
                     * in a call to update_beforeCurrent() on the next time step.
                     */
                    if(laserProfiles::Selected::INIT_TIME > 0.0_X)
                        LaserPhysics{}(currentStep);

                    // Incident field solver update does not use exchanged E, so does not have to wait for it
                    auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                    // update B by half step, to step currentStep + 1.5, so step for E_inc = currentStep + 1
                    incidentFieldSolver.updateBHalf(static_cast<float_X>(currentStep) + 1.0_X);

                    auto& fieldE = solver.getFieldE();
                    EventTask eRfieldE = fieldE.asyncCommunication(__getTransactionEvent());

                    solver.template updateBFirstHalf<CORE>(currentStep);
                    __setTransactionEvent(eRfieldE);
                    solver.template updateBFirstHalf<BORDER>(currentStep);

                    auto& fieldB = solver.getFieldB();
                    EventTask eRfieldB = fieldB.asyncCommunication(__getTransactionEvent());
                    __setTransactionEvent(eRfieldB);
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "Yee");
                    return propList;
                }

            private:
                MappingDesc const cellDescription;
                absorber::pml::Solver<CurlE, CurlB> solver;
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
