/**
 * Copyright 2014 Marco Garten
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

#include "simulation_defines.hpp"
#include "particles/memory/frames/Frame.hpp"
#include "traits/GetFlagType.hpp"
#include "traits/Resolve.hpp"
#include "simulation_defines/param/speciesDefinition.param"
#include "simulation_defines/unitless/speciesDefinition.unitless"

namespace picongpu
{

template<typename T_Species>
struct GetIonizer
{
    
    typedef typename T_Species::FrameType FrameType;

    typedef typename HasFlag<FrameType, ionizer<> >::type hasIonizer;
    
    /* The following line only fetches the alias */
    typedef typename GetFlagType<FrameType,ionizer<> >::type FoundIonizerAlias;
    /* This now resolves the alias into the actual object type */
    typedef typename PMacc::traits::Resolve<FoundIonizerAlias>::type FoundIonizer;
    /* if no ionizer was defined we use IonizerNone as fallback */
    typedef typename bmpl::if_<hasIonizer,FoundIonizer,particles::ionization::None >::type type;
    
};

}// namespace picongpu

