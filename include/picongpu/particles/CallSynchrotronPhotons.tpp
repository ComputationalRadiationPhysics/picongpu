/* Copyright 2014-2021 Rene Widera, Marco Garten, Alexander Grund,
 *                     Heiko Burau, Axel Huebl
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

#include "picongpu/fields/Fields.def"
#include "picongpu/particles/creation/creation.hpp"
#include "picongpu/particles/flylite/IFlyLite.hpp"
#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.hpp"
#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.tpp"
#include "picongpu/particles/traits/GetIonizerList.hpp"
#include "picongpu/particles/traits/GetPhotonCreator.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/communication/AsyncCommunication.hpp>
#include <pmacc/math/MapTuple.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/particles/traits/ResolveAliasFromSpecies.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/plus.hpp>

#include <memory>


namespace picongpu
{
    namespace particles
    {
/** Handles the synchrotron radiation emission of photons from electrons
         *
         * @tparam T_ElectronSpecies type or name as boost::mpl::string of electron particle species
         */
        template<typename T_ElectronSpecies>
        struct CallSynchrotronPhotons
        {
            using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;
            using ElectronFrameType = typename ElectronSpecies::FrameType;

            /* SelectedPhotonCreator will be either PhotonCreator or fallback: CreatorBase */
            using SelectedPhotonCreator = typename traits::GetPhotonCreator<ElectronSpecies>::type;
            using PhotonSpecies = typename SelectedPhotonCreator::PhotonSpecies;
            using PhotonFrameType = typename PhotonSpecies::FrameType;

            /** Functor implementation
             *
             * @tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * @param cellDesc logical block information like dimension and cell sizes
             * @param currentStep The current time step
             * @param synchrotronFunctions synchrotron functions wrapper object
             */
            template<typename T_CellDescription>
            HINLINE void operator()(
                T_CellDescription cellDesc,
                const uint32_t currentStep,
                const synchrotronPhotons::SynchrotronFunctions& synchrotronFunctions) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                /* alias for pointer on source species */
                auto electronSpeciesPtr = dc.get<ElectronSpecies>(ElectronFrameType::getName(), true);
                /* alias for pointer on destination species */
                auto photonSpeciesPtr = dc.get<PhotonSpecies>(PhotonFrameType::getName(), true);

                using namespace synchrotronPhotons;
                SelectedPhotonCreator photonCreator(
                    synchrotronFunctions.getCursor(SynchrotronFunctions::first),
                    synchrotronFunctions.getCursor(SynchrotronFunctions::second));

                creation::createParticlesFromSpecies(*electronSpeciesPtr, *photonSpeciesPtr, photonCreator, cellDesc);
            }
        };

    } // namespace particles
} // namespace picongpu
