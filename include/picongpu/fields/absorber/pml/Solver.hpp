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
#include "picongpu/fields/absorber/pml/Field.hpp"
#include "picongpu/fields/absorber/pml/Parameters.hpp"
#include "picongpu/fields/absorber/pml/Solver.kernel"
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
        namespace absorber
        {
            namespace pml
            {
                /** Implementation of Yee + PML solver updates of E and B
                 *
                 * The original paper on this approach is J.A. Roden, S.D. Gedney.
                 * Convolution PML (CPML): An efficient FDTD implementation of the
                 * CFS - PML for arbitrary media. Microwave and optical technology
                 * letters. 27 (5), 334-339 (2000).
                 * https://doi.org/10.1002/1098-2760(20001205)27:5%3C334::AID-MOP14%3E3.0.CO;2-A
                 * Our implementation is based on a more detailed description in section
                 * 7.9 of the book A. Taflove, S.C. Hagness. Computational
                 * Electrodynamics. The Finite-Difference Time-Domain Method. Third
                 * Edition. Artech house, Boston (2005), referred to as
                 * [Taflove, Hagness].
                 *
                 * @tparam T_CurlE functor to compute curl of E
                 * @tparam T_CurlB functor to compute curl of B
                 */
                template<typename T_CurlE, typename T_CurlB>
                class Solver
                {
                public:
                    using CurlE = T_CurlE;
                    using CurlB = T_CurlB;

                    Solver(MappingDesc const cellDescription) : cellDescription{cellDescription}
                    {
                        initParameters();
                        initFields();
                    }

                    //! Get a reference to field E
                    picongpu::FieldE& getFieldE()
                    {
                        return *(fieldE.get());
                    }

                    //! Get a reference to field B
                    picongpu::FieldB& getFieldB()
                    {
                        return *(fieldB.get());
                    }

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

                    /** Propagate E values in the given area by a time step.
                     *
                     * @tparam T_Area area to apply updates to, the curl must be
                     * applicable to all points; normally CORE, BORDER, or CORE + BORDER
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
                        using Kernel = KernelUpdateE<numWorkers, BlockDescription<CurlB>>;
                        AreaMapper<T_Area> mapper{cellDescription};
                        // Note: optimization considerations same as in updateBHalf( ).
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(), numWorkers)(
                            mapper,
                            getLocalParameters(mapper, currentStep),
                            CurlB(),
                            fieldB->getDeviceDataBox(),
                            fieldE->getDeviceDataBox(),
                            psiE->getDeviceOuterLayerBox());
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

                    // Yee solver data
                    std::shared_ptr<picongpu::FieldE> fieldE;
                    std::shared_ptr<picongpu::FieldB> fieldB;
                    MappingDesc cellDescription;

                    /* PML convolutional field data, defined as in [Taflove, Hagness],
                     * eq. (7.105a,b), and similar for other components
                     */
                    std::shared_ptr<pml::FieldE> psiE;
                    std::shared_ptr<pml::FieldB> psiB;

                    /** Thickness in terms of the global domain.
                     *
                     * We store only global thickness, as the local one can change
                     * during the simulation and so has to be recomputed for each time
                     * step. PML must be fully contained in a single layer of local
                     * domains near the global simulation area boundary. (Note that
                     * the domains of this layer might be changing, e.g. due to moving
                     * window.) There are no other limitations on PML thickness. In
                     * particular, it is independent of the BORDER area size.
                     */
                    Thickness globalSize;
                    Parameters parameters;

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
                        using Kernel = KernelUpdateBHalf<numWorkers, BlockDescription<CurlE>>;
                        AreaMapper<T_Area> mapper{cellDescription};
                        /* Note: here it is possible to first check if PML is enabled
                         * in the local domain at all, and otherwise optimize by calling
                         * the normal Yee update kernel. We do not do that, as this
                         * would be fragile with respect to future separation of PML
                         * into a plugin.
                         */
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(), numWorkers)(
                            mapper,
                            getLocalParameters(mapper, currentStep),
                            CurlE(),
                            fieldE->getDeviceDataBox(),
                            updatePsiB,
                            fieldB->getDeviceDataBox(),
                            psiB->getDeviceOuterLayerBox());
                    }

                    void initParameters()
                    {
                        namespace pml = maxwellSolver::Pml;

                        auto& absorber = absorber::Absorber::get();
                        globalSize = absorber.getGlobalThickness();
                        parameters.sigmaKappaGradingOrder = pml::SIGMA_KAPPA_GRADING_ORDER;
                        parameters.alphaGradingOrder = pml::ALPHA_GRADING_ORDER;
                        for(uint32_t dim = 0u; dim < simDim; dim++)
                        {
                            parameters.normalizedSigmaMax[dim] = pml::NORMALIZED_SIGMA_MAX[dim];
                            parameters.kappaMax[dim] = pml::KAPPA_MAX[dim];
                            parameters.normalizedAlphaMax[dim] = pml::NORMALIZED_ALPHA_MAX[dim];
                        }
                    }

                    void initFields()
                    {
                        /* Split fields are created here (and not with normal E and B)
                         * in order to not waste memory in case PML is not used.
                         */
                        DataConnector& dc = Environment<>::get().DataConnector();
                        fieldE = dc.get<picongpu::FieldE>(picongpu::FieldE::getName(), true);
                        fieldB = dc.get<picongpu::FieldB>(picongpu::FieldB::getName(), true);
                        psiE = std::make_shared<pml::FieldE>(cellDescription, globalSize);
                        psiB = std::make_shared<pml::FieldB>(cellDescription, globalSize);
                        dc.share(psiE);
                        dc.share(psiB);
                    }

                    template<uint32_t T_Area>
                    LocalParameters getLocalParameters(AreaMapper<T_Area>& mapper, uint32_t const currentStep) const
                    {
                        Thickness localThickness = getLocalThickness(currentStep);
                        checkLocalThickness(localThickness);
                        return LocalParameters(
                            parameters,
                            localThickness,
                            mapper.getGridSuperCells() * SuperCellSize::toRT(),
                            mapper.getGuardingSuperCells() * SuperCellSize::toRT());
                    }

                    /**
                     * Get PML thickness for the local domain at the current time step.
                     * It depends on the current step because of the moving window.
                     */
                    Thickness getLocalThickness(uint32_t const currentStep) const
                    {
                        /* The logic of the following checks is the same as in
                         * absorber::ExponentialDamping::run( ), to disable the absorber
                         * at a border we set the corresponding thickness to 0.
                         */
                        auto& movingWindow = MovingWindow::getInstance();
                        auto const numSlides = movingWindow.getSlideCounter(currentStep);
                        auto const numExchanges = NumberOfExchanges<simDim>::value;
                        auto const communicationMask
                            = Environment<simDim>::get().GridController().getCommunicationMask();
                        Thickness localThickness = globalSize;
                        for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
                        {
                            /* Here we are only interested in the positive and negative
                             * directions for x, y, z axes and not the "diagonal" ones.
                             * So skip other directions except left, right, top, bottom,
                             * back, front
                             */
                            if(FRONT % exchange != 0)
                                continue;

                            // Transform exchange into a pair of axis and direction
                            uint32_t axis = 0;
                            if(exchange >= BOTTOM && exchange <= TOP)
                                axis = 1;
                            if(exchange >= BACK)
                                axis = 2;
                            uint32_t direction = exchange % 2;

                            // No PML at the borders between two local domains
                            bool hasNeighbour = communicationMask.isSet(exchange);
                            if(hasNeighbour)
                                localThickness(axis, direction) = 0;

                            // Disable PML during laser initialization
                            if(fields::laserProfiles::Selected::initPlaneY == 0)
                            {
                                bool isLaserInitializationOver
                                    = (currentStep * DELTA_T) >= fields::laserProfiles::Selected::INIT_TIME;
                                if(numSlides == 0 && !isLaserInitializationOver && exchange == TOP)
                                    localThickness(axis, direction) = 0;
                            }

                            // Disable PML at the far side of the moving window
                            if(movingWindow.isSlidingWindowActive(currentStep) && exchange == BOTTOM)
                                localThickness(axis, direction) = 0;
                        }
                        return localThickness;
                    }

                    //! Verify that PML fits the local domain
                    void checkLocalThickness(Thickness const localThickness) const
                    {
                        auto const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                        auto const localPMLSize
                            = localThickness.getNegativeBorder() + localThickness.getPositiveBorder();
                        auto pmlFitsDomain = true;
                        for(uint32_t dim = 0u; dim < simDim; dim++)
                            if(localPMLSize[dim] > localDomain.size[dim])
                                pmlFitsDomain = false;
                        if(!pmlFitsDomain)
                            throw std::out_of_range("Requested PML size exceeds the local domain");
                    }

                    //! Get number of workers for kernels
                    static constexpr uint32_t getNumWorkers()
                    {
                        return pmacc::traits::GetNumWorkers<
                            pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                    }
                };

            } // namespace pml
        } // namespace absorber
    } // namespace fields
} // namespace picongpu

#include "picongpu/fields/absorber/pml/Field.tpp"
