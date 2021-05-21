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

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/LaserPhysics.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Yee.kernel"
#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/absorber/pml/Pml.hpp"
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
            // TODO move this comment
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
            class Yee
            {
            public:
                // Types required by field solver interface
                using CellType = cellType::Yee;
                using CurlE = T_CurlE;
                using CurlB = T_CurlB;

                Yee(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    fieldE = dc.get<FieldE>(FieldE::getName(), true);
                    fieldB = dc.get<FieldB>(FieldB::getName(), true);
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
                    updateBSecondHalf<CORE + BORDER>(currentStep);
                    auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                    // update B by half step, to step = currentStep + 0.5, so step for E_inc = currentStep
                    incidentFieldSolver.updateBHalf(static_cast<float_X>(currentStep));
                    EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());

                    updateE<CORE>(currentStep);
                    // Incident field solver update does not use exchanged B, so does not have to wait for it
                    // update E by full step, to step = currentStep + 1, so step for B_inc = currentStep + 0.5
                    incidentFieldSolver.updateE(static_cast<float_X>(currentStep) + 0.5_X);
                    __setTransactionEvent(eRfieldB);
                    updateE<BORDER>(currentStep);
                }

                /** Perform the last part of E and B propagation by a time step
                 *
                 * Together with update_beforeCurrent( ) forms the full propagation.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_afterCurrent(uint32_t const currentStep)
                {
                    /* As explained in more detail in comments inside update_beforeCurrent(), here we start a new step
                     * of updating B in terms of the time-staggered Yee grid. And so this is the first half of B
                     * update, to be complete in a call to update_beforeCurrent() on the next time step.
                     */
                    auto& absorber = absorber::Absorber::get();
                    absorber.run(currentStep, fieldE->getDeviceDataBox());
                    if(laserProfiles::Selected::INIT_TIME > 0.0_X)
                        LaserPhysics{}(currentStep);

                    // Incident field solver update does not use exchanged E, so does not have to wait for it
                    auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                    // update B by half step, to step currentStep + 1.5, so step for E_inc = currentStep + 1
                    incidentFieldSolver.updateBHalf(static_cast<float_X>(currentStep) + 1.0_X);

                    EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());

                    updateBFirstHalf<CORE>(currentStep);
                    __setTransactionEvent(eRfieldE);
                    updateBFirstHalf<BORDER>(currentStep);

                    absorber.run(currentStep, fieldB->getDeviceDataBox());

                    EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
                    __setTransactionEvent(eRfieldB);
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "Yee");
                    return propList;
                }

            private:
                // Helper types for configuring kernels
                template<typename T_Curl>
                using BlockDescription = pmacc::SuperCellDescription<
                    SuperCellSize,
                    typename traits::GetLowerMargin<T_Curl>::type,
                    typename traits::GetUpperMargin<T_Curl>::type>;
                template<uint32_t T_Area>
                using AreaMapper = pmacc::AreaMapping<T_Area, MappingDesc>;

                /** Propagate B values in the given area by the first half of a time step
                 *
                 * This operation propagates grid values of field B by dt/2 and prepares the internal state of
                 * convolutional components so that calling updateBSecondHalf() afterwards competes the update.
                 *
                 * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                 * normally CORE, BORDER, or CORE + BORDER
                 *
                 * @param currentStep index of the current time iteration
                 */
                template<uint32_t T_Area>
                void updateBFirstHalf(uint32_t const currentStep)
                {
                    updateBHalf<T_Area>(currentStep, true);
                }

                /** Propagate B values in the given area by the second half of a time step
                 *
                 * This operation propagates grid values of field B by dt/2 and relies on the internal state of
                 * convolutional components set up by a prior call to updateBFirstHalf(). After this call is
                 * completed, the convolutional components are in the state to call updateBFirstHalf() for the
                 * next time step.
                 *
                 * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                 * normally CORE, BORDER, or CORE + BORDER
                 *
                 * @param currentStep index of the current time iteration
                 */
                template<uint32_t T_Area>
                void updateBSecondHalf(uint32_t const currentStep)
                {
                    updateBHalf<T_Area>(currentStep, false);
                }

                /** Propagate B values in the given area by half a time step
                 *
                 * @tparam T_Area area to apply updates to, the curl must be
                 * applicable to all points; normally CORE, BORDER, or CORE + BORDER
                 *
                 * @param currentStep index of the current time iteration
                 * @param updatePsiB whether convolutional magnetic fields need to be updated, or are
                 * up-to-date
                 */
                template<uint32_t T_Area>
                void updateBHalf(uint32_t const currentStep, bool const updatePsiB)
                {
                    constexpr auto numWorkers = getNumWorkers();
                    using Kernel = yee::KernelUpdateB<numWorkers, BlockDescription<CurlE>>;
                    AreaMapper<T_Area> mapper{cellDescription};

                    // The ugly transition from run-time to compile-time polymorphism is contained here
                    auto& absorber = absorber::Absorber::get();
                    if(absorber.getKind() == absorber::Absorber::Kind::Pml)
                    {
                        auto& pmlInstance = absorber.asPml();
                        auto const updateFunctor
                            = pmlInstance.getUpdateBHalfFunctor<CurlE, T_Area>(currentStep, updatePsiB);
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(),
                         numWorkers)(mapper, updateFunctor, fieldE->getDeviceDataBox(), fieldB->getDeviceDataBox());
                    }
                    else
                        PMACC_KERNEL(Kernel{})
                    (mapper.getGridDim(), numWorkers)(
                        mapper,
                        yee::UpdateBHalfFunctor<CurlE>{},
                        fieldE->getDeviceDataBox(),
                        fieldB->getDeviceDataBox());
                }

                /** Propagate E values in the given area by a time step.
                 *
                 * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                 *                normally CORE, BORDER, or CORE + BORDER
                 *
                 * @param currentStep index of the current time iteration
                 */
                template<uint32_t T_Area>
                void updateE(uint32_t currentStep)
                {
                    /* Courant-Friedrichs-Levy-Condition for Yee Field Solver:
                     *
                     * A workaround is to add a template dependency to the expression.
                     * `sizeof(ANY_TYPE*) != 0` is always true and defers the evaluation.
                     */
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Levy_condition_failure____check_your_grid_param_file,
                        (SPEED_OF_LIGHT * SPEED_OF_LIGHT * DELTA_T * DELTA_T * INV_CELL2_SUM) <= 1.0
                            && sizeof(T_CurlE*) != 0);

                    constexpr auto numWorkers = getNumWorkers();
                    using Kernel = yee::KernelUpdateE<numWorkers, BlockDescription<CurlB>>;
                    auto mapper = AreaMapper<T_Area>{cellDescription};

                    // The ugly transition from run-time to compile-time polymorphism is contained here
                    auto& absorber = absorber::Absorber::get();
                    if(absorber.getKind() == absorber::Absorber::Kind::Pml)
                    {
                        auto& pmlInstance = absorber.asPml();
                        auto const updateFunctor = pmlInstance.getUpdateEFunctor<CurlB, T_Area>(currentStep);
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(),
                         numWorkers)(mapper, updateFunctor, fieldB->getDeviceDataBox(), fieldE->getDeviceDataBox());
                    }
                    else
                        PMACC_KERNEL(Kernel{})
                    (mapper.getGridDim(), numWorkers)(
                        mapper,
                        yee::UpdateEFunctor<CurlB>{},
                        fieldB->getDeviceDataBox(),
                        fieldE->getDeviceDataBox());
                }

                //! Get number of workers for kernels
                static constexpr uint32_t getNumWorkers()
                {
                    return pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                }

                MappingDesc const cellDescription;
                std::shared_ptr<FieldE> fieldE;
                std::shared_ptr<FieldB> fieldB;
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for B field access in the Yee solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::Yee<T_CurlE, T_CurlB>, FieldB>
        {
            using LowerMargin = typename T_CurlB::LowerMargin;
            using UpperMargin = typename T_CurlB::UpperMargin;
        };

        /** Get margin for E field access in the Yee solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::Yee<T_CurlE, T_CurlB>, FieldE>
        {
            using LowerMargin = typename T_CurlE::LowerMargin;
            using UpperMargin = typename T_CurlE::UpperMargin;
        };

        /** Get margin for both fields access in the Yee solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::Yee<T_CurlE, T_CurlB>>
        {
        private:
            using Solver = fields::maxwellSolver::Yee<T_CurlE, T_CurlB>;

        public:
            using LowerMargin = typename pmacc::math::CT::max<
                typename GetLowerMargin<Solver, FieldB>::type,
                typename GetLowerMargin<Solver, FieldE>::type>::type;
            using UpperMargin = typename pmacc::math::CT::max<
                typename GetUpperMargin<Solver, FieldB>::type,
                typename GetUpperMargin<Solver, FieldE>::type>::type;
        };

    } // namespace traits

} // namespace picongpu
