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

/* PIConGPU */
#include "picongpu/particles/flylite/NonLTE.hpp"
#include "picongpu/particles/flylite/helperFields/LocalEnergyHistogram.hpp"
#include "picongpu/particles/flylite/helperFields/LocalEnergyHistogramFunctors.hpp"
#include "picongpu/particles/flylite/helperFields/LocalRateMatrix.hpp"
#include "picongpu/particles/flylite/helperFields/LocalDensity.hpp"
#include "picongpu/particles/flylite/helperFields/LocalDensityFunctors.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Density.def"
#include "picongpu/particles/traits/GetShape.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <memory>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            template<
                //! @todo for multi ion species IPD: typename T_OtherIonsList

                typename T_ElectronsList,
                typename T_PhotonsList>
            void NonLTE<T_ElectronsList, T_PhotonsList>::init(
                pmacc::DataSpace<simDim> const& gridSizeLocal,
                std::string const& ionSpeciesName)
            {
                //! GPU-local number of cells in regular resolution (like FieldE & B)
                pmacc::DataSpace<simDim> m_gridSizeLocal = gridSizeLocal;
                //! GPU-local number of cells in averaged (reduced) resolution
                pmacc::DataSpace<simDim> m_avgGridSizeLocal
                    = m_gridSizeLocal / picongpu::flylite::spatialAverageBox::toRT();

                DataConnector& dc = Environment<>::get().DataConnector();

                // once allocated for all ion species to share
                if(!dc.hasId(helperFields::LocalEnergyHistogram::getName("electrons")))
                    dc.consume(std::make_unique<helperFields::LocalEnergyHistogram>("electrons", m_avgGridSizeLocal));

                if(!dc.hasId(helperFields::LocalEnergyHistogram::getName("photons")))
                    dc.consume(std::make_unique<helperFields::LocalEnergyHistogram>("photons", m_avgGridSizeLocal));

                if(!dc.hasId(helperFields::LocalDensity::getName("electrons")))
                    dc.consume(std::make_unique<helperFields::LocalDensity>("electrons", m_avgGridSizeLocal));

                // for each ion species
                if(!dc.hasId(helperFields::LocalRateMatrix::getName(ionSpeciesName)))
                    dc.consume(std::make_unique<helperFields::LocalRateMatrix>(ionSpeciesName, m_avgGridSizeLocal));

                if(!dc.hasId(helperFields::LocalDensity::getName(ionSpeciesName)))
                    dc.consume(std::make_unique<helperFields::LocalDensity>(ionSpeciesName, m_avgGridSizeLocal));
            }

            template<
                //! @todo for multi ion species IPD: typename T_OtherIonsList,

                typename T_ElectronsList,
                typename T_PhotonsList>
            template<typename T_IonSpeciesType>
            void NonLTE<
                //! @todo for multi ion species IPD: T_OtherIonsList,

                T_ElectronsList,
                T_PhotonsList>::update(std::string const& ionSpeciesName, uint32_t currentStep)
            {
                using IonSpeciesType = T_IonSpeciesType;

                // calculate density fields and energy histograms
                fillHelpers<IonSpeciesType>(ionSpeciesName, currentStep);

                //! @todo calculate rate matrix
                //! @todo implicit ODE solve to evolve populations
                //! @todo modify f_e of free electrons
                //! @todo modify f_ph of photon field (absorb)
                //! @todo change charges, create electrons & photons
            }

            template<
                //! @todo for multi ion species IPD: typename T_OtherIonsList,

                typename T_ElectronsList,
                typename T_PhotonsList>
            template<typename T_IonSpeciesType>
            void NonLTE<
                //! @todo for multi ion species IPD: T_OtherIonsList,

                T_ElectronsList,
                T_PhotonsList>::fillHelpers(std::string const& ionSpeciesName, uint32_t currentStep)
            {
                using IonSpeciesType = T_IonSpeciesType;

                // calculate density fields
                helperFields::FillLocalDensity<MakeSeq_t<IonSpeciesType>> fillDensityIons{};
                fillDensityIons(currentStep, ionSpeciesName);

                helperFields::FillLocalDensity<T_ElectronsList> fillDensityElectrons{};
                fillDensityElectrons(currentStep, "electrons");

                // calculate energy histograms: f(e), f(ph)
                helperFields::FillLocalEnergyHistogram<T_ElectronsList> fillEnergyHistogramElectrons{};
                fillEnergyHistogramElectrons(
                    currentStep,
                    "electrons",
                    picongpu::flylite::electronMinEnergy,
                    picongpu::flylite::electronMaxEnergy);

                helperFields::FillLocalEnergyHistogram<T_PhotonsList> fillEnergyHistogramPhotons{};
                fillEnergyHistogramPhotons(
                    currentStep,
                    "photons",
                    picongpu::flylite::photonMinEnergy,
                    picongpu::flylite::photonMaxEnergy);
            }

        } // namespace flylite
    } // namespace particles
} // namespace picongpu
