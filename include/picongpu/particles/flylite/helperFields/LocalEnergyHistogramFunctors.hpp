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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/flylite/helperFields/LocalEnergyHistogram.hpp"
#include "picongpu/particles/flylite/helperFields/LocalEnergyHistogram.kernel"

// pmacc
#include <pmacc/static_assert.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/meta/ForEach.hpp>

#include <string>
#include <memory>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            namespace helperFields
            {
                namespace detail
                {
                    /** Takes a single species and adds it to a LocalEnergyHistogram
                     *
                     * @tparam T_SpeciesType a picongpu::Particles class with a particle species
                     */
                    template<typename T_SpeciesType>
                    struct AddSingleEnergyHistogram
                    {
                        using SpeciesType = T_SpeciesType;
                        using FrameType = typename SpeciesType::FrameType;

                        /** Functor
                         *
                         * @param currentStep the current time step
                         * @param eneHistLocal the GridBuffer for local energy histograms
                         * @param minEnergy minimum energy to account for (eV)
                         * @param maxEnergy maximum energy to account for (eV)
                         */
                        void operator()(
                            uint32_t currentStep,
                            std::shared_ptr<LocalEnergyHistogram>& eneHistLocal,
                            float_X const minEnergy,
                            float_X const maxEnergy)
                        {
                            DataConnector& dc = Environment<>::get().DataConnector();

                            // load particle without copy particle data to host
                            auto speciesTmp = dc.get<SpeciesType>(FrameType::getName(), true);

                            // mapper to access species in CORE & BORDER only
                            MappingDesc cellDescription(
                                speciesTmp->getParticlesBuffer().getSuperCellsLayout().getDataSpace()
                                    * SuperCellSize::toRT(),
                                GuardSize::toRT());
                            AreaMapping<CORE + BORDER, MappingDesc> mapper(cellDescription);

                            // add energy histogram on top of existing data
                            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                                pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                            PMACC_KERNEL(helperFields::KernelAddLocalEnergyHistogram<numWorkers>{})
                            (
                                // one block per local energy histogram
                                mapper.getGridDim(),
                                numWorkers)(
                                // start in border (jump over GUARD area)
                                speciesTmp->getDeviceParticlesBox(),
                                // start in border (has no GUARD area)
                                eneHistLocal->getGridBuffer().getDeviceBuffer().getDataBox(),
                                minEnergy,
                                maxEnergy,
                                mapper);

                            dc.releaseData(FrameType::getName());
                        }
                    };
                } // namespace detail
                /** Add a group of species to a local energy histogram
                 *
                 * Takes a list of species and fills the LocalEnergyHistogram with it.
                 * Ideally executed for a list of electron species or an photon species.
                 *
                 * @tparam T_SpeciesList sequence of picongpu::Particles to create a
                 *                       local energy histogram from
                 */
                template<typename T_SpeciesList>
                struct FillLocalEnergyHistogram
                {
                    using SpeciesList = T_SpeciesList;

                    /** Functor
                     *
                     * @param currentStep the current time step
                     * @param speciesGroup naming for the group of species in T_SpeciesList
                     * @param minEnergy minimum energy to account for (eV)
                     * @param maxEnergy maximum energy to account for (eV)
                     */
                    void operator()(
                        uint32_t currentStep,
                        std::string const& speciesGroup,
                        float_X const minEnergy,
                        float_X const maxEnergy)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();

                        /* load local energy histogram field without copy data to host and
                         * zero it
                         */
                        auto eneHistLocal = dc.get<LocalEnergyHistogram>(
                            helperFields::LocalEnergyHistogram::getName(speciesGroup),
                            true);

                        // reset local energy histograms
                        eneHistLocal->getGridBuffer().getDeviceBuffer().setValue(float_X(0.0));

                        // add local energy histogram of each species in list
                        meta::ForEach<SpeciesList, detail::AddSingleEnergyHistogram<bmpl::_1>>
                            addSingleEnergyHistogram;
                        addSingleEnergyHistogram(currentStep, eneHistLocal, minEnergy, maxEnergy);

                        /* note: for average != supercell the BORDER region would need to be
                         *       build up via communication accordingly
                         */

                        // release fields
                        dc.releaseData(helperFields::LocalEnergyHistogram::getName(speciesGroup));
                    }
                };

            } // namespace helperFields
        } // namespace flylite
    } // namespace particles
} // namespace picongpu
