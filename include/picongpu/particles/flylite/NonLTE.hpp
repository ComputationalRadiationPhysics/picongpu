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

#include "picongpu/particles/flylite/NonLTE.def"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/flylite/IFlyLite.hpp"

/* pmacc */
#include <pmacc/dimensions/DataSpace.hpp>

#include <memory>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            template<
                //! @todo for multi ion species IPD: typename T_OtherIonsList,
                typename T_ElectronsList,
                typename T_PhotonsList>
            class NonLTE : public IFlyLite
            {
            public:
                //! @todo for multi ion species IPD: using OtherIonsList = T_OtherIonsList;

                using ElectronsList = T_ElectronsList;
                using PhotonsList = T_PhotonsList;

                virtual void init(pmacc::DataSpace<simDim> const& gridSizeLocal, std::string const& ionSpeciesName);

                /** Update atomic configurations
                 *
                 * Prepares auxiliary fields for the non-LTE atomic physics model and
                 * updates the configurations & charge states of an ion species.
                 *
                 * @tparam T_IonSpeciesType a picongpu::Particles class with an ion
                 *                          species
                 *
                 * @param ionSpeciesName unique name of the ion species in T_IonSpeciesType
                 * @param currentStep the current time step
                 */
                template<typename T_IonSpeciesType>
                void update(std::string const& ionSpeciesName, uint32_t currentStep);

            private:
                /** Calculate new values in helper fields
                 *
                 * Prepares helper fields by calculating local densities and energy
                 * histograms.
                 *
                 * @param ionSpeciesName unique name of the ion species in T_IonSpeciesType
                 * @param currentStep the current time step
                 */
                template<typename T_IonSpeciesType>
                void fillHelpers(std::string const& ionSpeciesName, uint32_t currentStep);
            };

        } // namespace flylite
    } // namespace particles
} // namespace picongpu
