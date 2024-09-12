/* Copyright 2019-2023 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/MaxwellSolver/AddCurrentDensity.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTDBase.kernel"
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/absorber.hpp"
#include "picongpu/fields/absorber/pml/Pml.hpp"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
#include "picongpu/fields/incidentField/Solver.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include <cstdint>
#include <memory>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace fdtd
            {
                /** Base class with common implementation details for finite-difference time-domain field solvers
                 *
                 * All field update operations are expressed in terms of getTimeStep().
                 * They work for both getTimeStep() == sim.pic.getDt() and getTimeStep() != sim.pic.getDt().
                 * Thus, they can be used to construct both a normal and a substepping version of FDTD.
                 *
                 * @tparam T_CurlE functor to compute curl of E
                 * @tparam T_CurlB functor to compute curl of B
                 */
                template<typename T_CurlE, typename T_CurlB>
                class FDTDBase
                {
                public:
                    //! Curl(E) functor type
                    using CurlE = T_CurlE;

                    //! Curl(B) functor type
                    using CurlB = T_CurlB;

                    /** Create FDTD base solver instance
                     *
                     * @param cellDescription mapping description for kernels
                     */
                    FDTDBase(MappingDesc const cellDescription)
                        : cellDescription(cellDescription)
                        ,
                        // Make sure the absorber instance is created here, before particle memory allocation
                        absorberImpl(fields::absorber::AbsorberImpl::getImpl(cellDescription))
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        fieldE = dc.get<FieldE>(FieldE::getName());
                        fieldB = dc.get<FieldB>(FieldB::getName());
                    }

                protected:
                    //! Time step used for each field solver update
                    static constexpr auto timeStep = getTimeStep();

                    /** Perform the first part of E and B propagation
                     *  from t_start = currentStep * sim.pic.getDt() to t_end = t_start + timeStep.
                     *
                     * Does not account for the J term, which will be added by addCurrent().
                     * Together with addCurrent() and updateAfterCurrent() forms the full propagation by timeStep.
                     *
                     * @param currentStep index of the current time iteration,
                     *                    note that it is in units of sim.pic.getDt(), not timeStep
                     */
                    void updateBeforeCurrent(float_X const currentStep)
                    {
                        /* As typical for electrodynamic PIC codes, we split an FDTD update of B into two halves.
                         * (This comes from commonly used particle pushers needing E and B at the same time.)
                         * Here we do the second half of updating B.
                         * It completes the first half in updateAfterCurrent() at the previous time (sub)step.
                         * In vacuum, the updates are linear and splitting could only influence floating-point
                         * arithmetic. For PML there is a difference, it is treated inside the absorber implementation.
                         * In both cases, the full E and B fields behave as expected after the update.
                         */
                        updateBSecondHalf<CORE + BORDER>(currentStep);
                        auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                        // update B by half timeStep, so step for E_inc = currentStep
                        incidentFieldSolver.updateBHalf(currentStep);
                        EventTask eRfieldB = fieldB->asyncCommunication(eventSystem::getTransactionEvent());
                        /* Incident field solver update does not use exchanged B, so does not have to wait for it.
                         * Update E by timeStep, to time = currentStep * sim.pic.getDt() + timeStep.
                         * It uses values of B_inc at time = currentStep * sim.pic.getDt() + 0.5 * timeStep.
                         * In units of sim.pic.getDt() that is equal to currentStep + 0.5 * timeStep / sim.pic.getDt()
                         */
                        incidentFieldSolver.updateE(currentStep + 0.5_X * timeStep / sim.pic.getDt());
                        updateE<CORE>(currentStep);
                        eventSystem::setTransactionEvent(eRfieldB);
                        updateE<BORDER>(currentStep);
                    }

                    /** Add contribution of the given current density with the given coefficient
                     *
                     * @tparam T_area area to operate on
                     * @tparam T_JBox type of device data box with current density values
                     *
                     * @param dataBoxJ device data box with current density values
                     * @param coeff coefficient value
                     */
                    template<uint32_t T_area, typename T_JBox>
                    void addCurrentImpl(T_JBox dataBoxJ, float_X const coeff)
                    {
                        auto const addCurrentDensity = AddCurrentDensity<T_area>{cellDescription};
                        auto const kind = currentInterpolation::CurrentInterpolation::get().kind;
                        if(kind == currentInterpolation::CurrentInterpolation::Kind::None)
                            addCurrentDensity(dataBoxJ, currentInterpolation::None{}, coeff);
                        else
                            addCurrentDensity(dataBoxJ, currentInterpolation::Binomial{}, coeff);
                    }

                    /** Perform the last part of E and B propagation
                     *  from t_start = currentStep * sim.pic.getDt() to t_end = t_start + timeStep.
                     *
                     * Does not account for the J term, which has been added by addCurrent().
                     * Together with addCurrent() and updateBeforeCurrent() forms the full propagation by timeStep.
                     *
                     * @param currentStep index of the current time iteration,
                     *                    note that it is in units of sim.pic.getDt(), not timeStep
                     */
                    void updateAfterCurrent(float_X const currentStep)
                    {
                        auto& absorber = absorber::Absorber::get();
                        if(absorber.getKind() == absorber::Absorber::Kind::Exponential)
                        {
                            auto& exponentialImpl = absorberImpl.asExponentialImpl();
                            exponentialImpl.run(currentStep, fieldE->getDeviceDataBox());
                        }

                        // Incident field solver update does not use exchanged E, so does not have to wait for it
                        auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                        /* Update B by half timeStep, to time = currentStep * sim.pic.getDt() + 1.5 * timeStep.
                         * It uses values of E_inc at time = currentStep * sim.pic.getDt() + timeStep.
                         * In units of sim.pic.getDt() that is equal to currentStep + timeStep / sim.pic.getDt()
                         */
                        incidentFieldSolver.updateBHalf(currentStep + timeStep / sim.pic.getDt());

                        EventTask eRfieldE = fieldE->asyncCommunication(eventSystem::getTransactionEvent());

                        // First and second halves of B update are explained inside updateBeforeCurrent()
                        updateBFirstHalf<CORE>(currentStep);
                        eventSystem::setTransactionEvent(eRfieldE);
                        updateBFirstHalf<BORDER>(currentStep);

                        if(absorber.getKind() == absorber::Absorber::Kind::Exponential)
                        {
                            auto& exponentialImpl = absorberImpl.asExponentialImpl();
                            exponentialImpl.run(currentStep, fieldB->getDeviceDataBox());
                        }

                        EventTask eRfieldB = fieldB->asyncCommunication(eventSystem::getTransactionEvent());
                        eventSystem::setTransactionEvent(eRfieldB);
                    }

                private:
                    // Helper types for configuring kernels
                    template<typename T_Curl>
                    using BlockDescription = pmacc::SuperCellDescription<
                        SuperCellSize,
                        typename picongpu::traits::GetLowerMargin<T_Curl>::type,
                        typename picongpu::traits::GetUpperMargin<T_Curl>::type>;

                    /** Propagate B values in the given area by the first half of a timeStep
                     *
                     * This operation propagates grid values of field B by timeStep/2.
                     * If PML is used, it also prepares the internal state of convolutional components
                     * so that calling updateBSecondHalf() afterwards competes the update.
                     *
                     * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                     *                normally CORE, BORDER, or CORE + BORDER
                     *
                     * @param currentStep index of the current time iteration
                     */
                    template<uint32_t T_Area>
                    void updateBFirstHalf(float_X const currentStep)
                    {
                        updateBHalf<T_Area>(currentStep, true);
                    }

                    /** Propagate B values in the given area by the second half of a timeStep
                     *
                     * This operation propagates grid values of field B by timeStep/2.
                     * If PML is used, it relies on the internal state of convolutional components
                     * having been set up by a prior call to updateBFirstHalf().
                     * Then this call leaves it ready for updateBFirstHalf() called on the next time step.
                     *
                     * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                     *                normally CORE, BORDER, or CORE + BORDER
                     *
                     * @param currentStep index of the current time iteration
                     */
                    template<uint32_t T_Area>
                    void updateBSecondHalf(float_X const currentStep)
                    {
                        updateBHalf<T_Area>(currentStep, false);
                    }

                    /** Propagate B values in the given area by half a timeStep
                     *
                     * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                     *                normally CORE, BORDER, or CORE + BORDER
                     *
                     * @param currentStep index of the current time iteration
                     * @param updatePsiB whether convolutional magnetic fields need to be updated, or are up-to-date
                     */
                    template<uint32_t T_Area>
                    void updateBHalf(float_X const currentStep, bool const updatePsiB)
                    {
                        using Kernel = fdtd::KernelUpdateField;
                        auto const mapper = pmacc::makeAreaMapper<T_Area>(cellDescription);

                        // The ugly transition from run-time to compile-time polymorphism is contained here
                        auto& absorber = absorber::Absorber::get();
                        if(absorber.getKind() == absorber::Absorber::Kind::Pml)
                        {
                            auto& pmlImpl = absorberImpl.asPmlImpl();
                            auto const updateFunctor
                                = pmlImpl.template getUpdateBHalfFunctor<CurlE>(currentStep, updatePsiB);
                            PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), SuperCellSize{})(
                                mapper,
                                updateFunctor,
                                fieldE->getDeviceDataBox(),
                                fieldB->getDeviceDataBox());
                        }
                        else
                        {
                            PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), SuperCellSize{})(
                                mapper,
                                fdtd::UpdateBHalfFunctor<CurlE>{},
                                fieldE->getDeviceDataBox(),
                                fieldB->getDeviceDataBox());
                        }
                    }

                    /** Propagate E values in the given area by a timeStep.
                     *
                     * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                     *                normally CORE, BORDER, or CORE + BORDER
                     *
                     * @param currentStep index of the current time iteration
                     */
                    template<uint32_t T_Area>
                    void updateE(float_X currentStep)
                    {
                        using Kernel = fdtd::KernelUpdateField;
                        auto const mapper = pmacc::makeAreaMapper<T_Area>(cellDescription);

                        // The ugly transition from run-time to compile-time polymorphism is contained here
                        auto& absorber = absorber::Absorber::get();
                        if(absorber.getKind() == absorber::Absorber::Kind::Pml)
                        {
                            auto& pmlImpl = absorberImpl.asPmlImpl();
                            auto const updateFunctor = pmlImpl.template getUpdateEFunctor<CurlB>(currentStep);
                            PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), SuperCellSize{})(
                                mapper,
                                updateFunctor,
                                fieldB->getDeviceDataBox(),
                                fieldE->getDeviceDataBox());
                        }
                        else
                        {
                            PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), SuperCellSize{})(
                                mapper,
                                fdtd::UpdateEFunctor<CurlB>{},
                                fieldB->getDeviceDataBox(),
                                fieldE->getDeviceDataBox());
                        }
                    }

                    MappingDesc const cellDescription;
                    std::shared_ptr<FieldE> fieldE;
                    std::shared_ptr<FieldB> fieldB;

                    // Absorber implementation
                    fields::absorber::AbsorberImpl& absorberImpl;
                };
            } // namespace fdtd
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
