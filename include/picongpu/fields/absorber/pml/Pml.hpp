/* Copyright 2019-2023 Sergei Bastrakov
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

#include "picongpu/fields/absorber/AbsorberImpl.hpp"
#include "picongpu/fields/absorber/pml/Field.hpp"
#include "picongpu/fields/absorber/pml/Parameters.hpp"
#include "picongpu/fields/absorber/pml/Pml.kernel"

#include <cstdint>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            namespace pml
            {
                /** Implementation of FDTD + PML updates of E and B
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
                 */
                class PmlImpl : public AbsorberImpl
                {
                public:
                    /** Create PML absorber implementation instance
                     *
                     * @param cellDescription mapping for kernels
                     */
                    PmlImpl(MappingDesc const cellDescription) : AbsorberImpl(Absorber::Kind::Pml, cellDescription)
                    {
                        initParameters();
                        DataConnector& dc = Environment<>::get().DataConnector();
                        psiE = std::make_shared<pml::FieldE>(cellDescription, getGlobalThickness());
                        psiB = std::make_shared<pml::FieldB>(cellDescription, getGlobalThickness());
                        dc.share(psiE);
                        dc.share(psiB);
                    }

                    /** Functor to update electric field by a time step using FDTD with the given curl and PML
                     *
                     * @tparam T_CurlB curl functor type according to the Curl concept
                     *
                     * @param currentStep index of the current time iteration
                     */
                    template<typename T_CurlB>
                    UpdateEFunctor<T_CurlB> getUpdateEFunctor(float_X const currentStep)
                    {
                        return UpdateEFunctor<T_CurlB>{
                            psiE->getDeviceOuterLayerBox(),
                            getLocalParameters(currentStep)};
                    }

                    /** Functor to update magnetic field by half a time step using FDTD with the given curl and PML
                     *
                     * @tparam T_CurlE curl functor type according to the Curl concept
                     *
                     * @param currentStep index of the current time iteration
                     * @param updatePsiB whether convolutional magnetic fields need to be updated, or are
                     * up-to-date
                     */
                    template<typename T_CurlE>
                    UpdateBHalfFunctor<T_CurlE> getUpdateBHalfFunctor(float_X const currentStep, bool const updatePsiB)
                    {
                        return UpdateBHalfFunctor<T_CurlE>{
                            psiB->getDeviceOuterLayerBox(),
                            getLocalParameters(currentStep),
                            updatePsiB};
                    }

                    /** Get parameters for the local domain
                     *
                     * @param currentStep index of the current time iteration
                     */
                    LocalParameters getLocalParameters(float_X const currentStep) const
                    {
                        Thickness localThickness = getLocalThickness(currentStep);
                        checkLocalThickness(localThickness);
                        return LocalParameters(
                            parameters,
                            localThickness,
                            cellDescription.getGridSuperCells() * SuperCellSize::toRT(),
                            cellDescription.getGuardingSuperCells() * SuperCellSize::toRT());
                    }

                private:
                    //! Read parameters from fieldAbsorber.param
                    void initParameters()
                    {
                        parameters.sigmaKappaGradingOrder = SIGMA_KAPPA_GRADING_ORDER;
                        parameters.alphaGradingOrder = ALPHA_GRADING_ORDER;
                        for(uint32_t dim = 0u; dim < simDim; dim++)
                        {
                            parameters.normalizedSigmaMax[dim] = NORMALIZED_SIGMA_MAX[dim];
                            parameters.kappaMax[dim] = KAPPA_MAX[dim];
                            parameters.normalizedAlphaMax[dim] = NORMALIZED_ALPHA_MAX[dim];
                        }
                    }

                    /** Get PML thickness for the local domain at the current time step.
                     *
                     * It depends on the current step because of the moving window.
                     */
                    Thickness getLocalThickness(float_X const currentStep) const
                    {
                        /* The logic of the following checks is to disable the absorber
                         * at a border we set the corresponding thickness to 0.
                         */
                        auto& movingWindow = MovingWindow::getInstance();
                        auto const numExchanges = NumberOfExchanges<simDim>::value;
                        auto const communicationMask
                            = Environment<simDim>::get().GridController().getCommunicationMask();
                        Thickness localThickness = getGlobalThickness();
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

                            // Disable PML at the far side of the moving window
                            if(movingWindow.isSlidingWindowActive(static_cast<uint32_t>(currentStep))
                               && exchange == BOTTOM)
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

                    //! Parameters
                    Parameters parameters;

                    /* PML convolutional field data, defined as in [Taflove, Hagness],
                     * eq. (7.105a,b), and similar for other components
                     */
                    std::shared_ptr<pml::FieldE> psiE;
                    std::shared_ptr<pml::FieldB> psiB;
                };

            } // namespace pml
        } // namespace absorber
    } // namespace fields
} // namespace picongpu

#include "picongpu/fields/absorber/pml/Field.tpp"
