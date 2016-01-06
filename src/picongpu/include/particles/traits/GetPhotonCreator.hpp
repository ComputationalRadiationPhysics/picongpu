/**
 * Copyright 2015 Heiko Burau
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

#include "types.h"
#include "simulation_defines.hpp"
#include "particles/memory/frames/Frame.hpp"
#include "traits/GetFlagType.hpp"
#include "traits/Resolve.hpp"
#include "simulation_defines/param/speciesDefinition.param"
#include "simulation_defines/unitless/speciesDefinition.unitless"

#include "particles/bremsstrahlung/PhotonCreator.def"
#include "particles/bremsstrahlung/PhotonCreator.hpp"

namespace picongpu
{

template<typename T_SpeciesType>
struct GetPhotonCreator
{

    typedef T_SpeciesType SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    typedef typename HasFlag<FrameType, bremsstrahlung_effect<> >::type hasBremsstrahlung;

    /* The following line only fetches the alias */
    typedef typename GetFlagType<FrameType, bremsstrahlung_effect<> >::type FoundBremsstrahlungAlias;

    /* This now resolves the alias into the actual object type */
    typedef typename PMacc::traits::Resolve<FoundBremsstrahlungAlias>::type FoundPhotonSpecies;

    /* This specifies the source species as the second template parameter of the photon creator */
    typedef particles::bremsstrahlung::PhotonCreator<T_SpeciesType, FoundPhotonSpecies> type;

}; // struct GetPhotonCreator

}// namespace picongpu
