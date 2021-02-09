/* Copyright 2015-2021 Heiko Burau
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
#include <pmacc/types.hpp>
#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include "picongpu/particles/synchrotronPhotons/PhotonCreator.def"


namespace picongpu
{
    namespace particles
    {
        namespace traits
        {
            /** Get the functor to create photons from a species
             *
             * @tparam T_SpeciesType type or name as boost::mpl::string
             */
            template<typename T_SpeciesType>
            struct GetPhotonCreator
            {
                using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
                using FrameType = typename SpeciesType::FrameType;

                // The following line only fetches the alias
                using FoundSynchrotronPhotonsAlias =
                    typename GetFlagType<FrameType, picongpu::synchrotronPhotons<>>::type;

                // This now resolves the alias into the actual object type and select the species from the species list
                using FoundPhotonSpecies = pmacc::particles::meta::FindByNameOrType_t<
                    VectorAllSpecies,
                    typename pmacc::traits::Resolve<FoundSynchrotronPhotonsAlias>::type>;

                // This specifies the target species as the second template parameter of the photon creator
                using type = synchrotronPhotons::PhotonCreator<SpeciesType, FoundPhotonSpecies>;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
