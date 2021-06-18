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
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.def"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.kernel"
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
            template<typename T_CurlE, typename T_CurlB>
            class FDTD
            {
            public:
                // Types required by field solver interface
                using CellType = cellType::Yee;
                using CurlE = T_CurlE;
                using CurlB = T_CurlB;

                FDTD(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                    CFLChecker<FDTD>{}();
                    DataConnector& dc = Environment<>::get().DataConnector();
                    fieldE = dc.get<FieldE>(FieldE::getName(), true);
                    fieldB = dc.get<FieldB>(FieldB::getName(), true);
                    auto& absorberFactory = fields::absorber::AbsorberFactory::get();
                    absorberImpl = absorberFactory.makeImpl(cellDescription);
                }

                /** Perform the first part of E and B propagation by a time step.
                 *
                 * Together with update_afterCurrent( ) forms the full propagation.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_beforeCurrent(uint32_t const currentStep)
                {
                    /* As typical for electrodynamic PIC codes, we split an FDTD update of B into two halves.
                     * (This comes from commonly used particle pushers needing E and B at the same time.)
                     * Here we do the second half of updating B.
                     * It completes the first half in update_afterCurrent() at the previous time step.
                     * In vacuum, the updates are linear and splitting could only influence floating-point arithmetic.
                     * For PML there is a difference, it is treated inside the absorber implementation.
                     * In both cases, the full E and B fields behave as expected after the update.
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
                    auto& absorber = absorber::Absorber::get();
                    if(absorber.getKind() == absorber::Absorber::Kind::Exponential)
                    {
                        auto& exponentialImpl = absorberImpl->asExponentialImpl();
                        exponentialImpl.run(currentStep, fieldE->getDeviceDataBox());
                    }
                    if(laserProfiles::Selected::INIT_TIME > 0.0_X)
                        LaserPhysics{}(currentStep);

                    // Incident field solver update does not use exchanged E, so does not have to wait for it
                    auto incidentFieldSolver = fields::incidentField::Solver{cellDescription};
                    // update B by half step, to step currentStep + 1.5, so step for E_inc = currentStep + 1
                    incidentFieldSolver.updateBHalf(static_cast<float_X>(currentStep) + 1.0_X);

                    EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());

                    // First and second halves of B update are explained inside update_beforeCurrent()
                    updateBFirstHalf<CORE>(currentStep);
                    __setTransactionEvent(eRfieldE);
                    updateBFirstHalf<BORDER>(currentStep);

                    if(absorber.getKind() == absorber::Absorber::Kind::Exponential)
                    {
                        auto& exponentialImpl = absorberImpl->asExponentialImpl();
                        exponentialImpl.run(currentStep, fieldB->getDeviceDataBox());
                    }

                    EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
                    __setTransactionEvent(eRfieldB);
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "FDTD");
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
                 * This operation propagates grid values of field B by dt/2.
                 * If PML is used, it also prepares the internal state of convolutional components
                 * so that calling updateBSecondHalf() afterwards competes the update.
                 *
                 * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                 *                normally CORE, BORDER, or CORE + BORDER
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
                 * This operation propagates grid values of field B by dt/2.
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
                void updateBSecondHalf(uint32_t const currentStep)
                {
                    updateBHalf<T_Area>(currentStep, false);
                }

                /** Propagate B values in the given area by half a time step
                 *
                 * @tparam T_Area area to apply updates to, the curl must be applicable to all points;
                 *                normally CORE, BORDER, or CORE + BORDER
                 *
                 * @param currentStep index of the current time iteration
                 * @param updatePsiB whether convolutional magnetic fields need to be updated, or are up-to-date
                 */
                template<uint32_t T_Area>
                void updateBHalf(uint32_t const currentStep, bool const updatePsiB)
                {
                    constexpr auto numWorkers = getNumWorkers();
                    using Kernel = fdtd::KernelUpdateField<numWorkers>;
                    AreaMapper<T_Area> mapper{cellDescription};

                    // The ugly transition from run-time to compile-time polymorphism is contained here
                    auto& absorber = absorber::Absorber::get();
                    if(absorber.getKind() == absorber::Absorber::Kind::Pml)
                    {
                        auto& pmlImpl = absorberImpl->asPmlImpl();
                        auto const updateFunctor
                            = pmlImpl.getUpdateBHalfFunctor<CurlE, T_Area>(currentStep, updatePsiB);
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(),
                         numWorkers)(mapper, updateFunctor, fieldE->getDeviceDataBox(), fieldB->getDeviceDataBox());
                    }
                    else
                    {
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(), numWorkers)(
                            mapper,
                            fdtd::UpdateBHalfFunctor<CurlE>{},
                            fieldE->getDeviceDataBox(),
                            fieldB->getDeviceDataBox());
                    }
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
                    constexpr auto numWorkers = getNumWorkers();
                    using Kernel = fdtd::KernelUpdateField<numWorkers>;
                    auto mapper = AreaMapper<T_Area>{cellDescription};

                    // The ugly transition from run-time to compile-time polymorphism is contained here
                    auto& absorber = absorber::Absorber::get();
                    if(absorber.getKind() == absorber::Absorber::Kind::Pml)
                    {
                        auto& pmlImpl = absorberImpl->asPmlImpl();
                        auto const updateFunctor = pmlImpl.getUpdateEFunctor<CurlB, T_Area>(currentStep);
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(),
                         numWorkers)(mapper, updateFunctor, fieldB->getDeviceDataBox(), fieldE->getDeviceDataBox());
                    }
                    else
                    {
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(), numWorkers)(
                            mapper,
                            fdtd::UpdateEFunctor<CurlB>{},
                            fieldB->getDeviceDataBox(),
                            fieldE->getDeviceDataBox());
                    }
                }

                //! Get number of workers for kernels
                static constexpr uint32_t getNumWorkers()
                {
                    return pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                }

                MappingDesc const cellDescription;
                std::shared_ptr<FieldE> fieldE;
                std::shared_ptr<FieldB> fieldB;

                // Absorber implementation
                std::unique_ptr<fields::absorber::AbsorberImpl> absorberImpl;
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for B field access in the FDTD solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>, FieldB>
        {
            using LowerMargin = typename T_CurlB::LowerMargin;
            using UpperMargin = typename T_CurlB::UpperMargin;
        };

        /** Get margin for E field access in the FDTD solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>, FieldE>
        {
            using LowerMargin = typename T_CurlE::LowerMargin;
            using UpperMargin = typename T_CurlE::UpperMargin;
        };

        /** Get margin for both fields access in the FDTD solver
         *
         * @tparam T_CurlE functor to compute curl of E
         * @tparam T_CurlB functor to compute curl of B
         */
        template<typename T_CurlE, typename T_CurlB>
        struct GetMargin<fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>>
        {
        private:
            using Solver = fields::maxwellSolver::FDTD<T_CurlE, T_CurlB>;

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
