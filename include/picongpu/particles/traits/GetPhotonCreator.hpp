/* Copyright 2015-2018 Heiko Burau
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

#include "picongpu/particles/synchrotronPhotons/PhotonCreator.def"

namespace picongpu
{
namespace particles
{
namespace traits
{

template<typename T_SpeciesType>
struct GetPhotonCreator
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    /* The following line only fetches the alias */
    typedef typename GetFlagType<FrameType, picongpu::synchrotronPhotons<> >::type FoundSynchrotronPhotonsAlias;

    /* This now resolves the alias into the actual object type */
    typedef typename pmacc::traits::Resolve<FoundSynchrotronPhotonsAlias>::type FoundPhotonSpecies;

    /* This specifies the target species as the second template parameter of the photon creator */
    typedef synchrotronPhotons::PhotonCreator<SpeciesType, FoundPhotonSpecies> type;

};

} // namespace traits
} // namespace particles
} // namespace picongpu
