/**
 * Copyright 2013 Axel Huebl, René Widera
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
 


#ifndef SPECIES_HPP
#define	SPECIES_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "particles/Particles.hpp"
#include "particles/species/ions/IonMethods.hpp"
#include "particles/species/electrons/ElectronMethods.hpp"
#include "particles/species/default/ParticlesData.hpp"

#include "plugins/radiation/parameters.hpp"
#if(ENABLE_RADIATION == 1)
#include "plugins/radiation/particles/Momentum_mt1.hpp"
#include "plugins/radiation/particles/RadiationFlag.hpp"
#endif

#include <boost/mpl/vector.hpp>

namespace picongpu
{
using namespace PMacc;


typedef Particles<
    typename bmpl::vector<
        IonMethods<>,
        #if(ENABLE_RADIATION == 1)
        Momentum_mt1<>,
        #if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
        RadiationFlag<>,
        #endif
        #endif
        ParticlesData<> 
    >::type
> PIC_Ions;

typedef Particles<
    typename  bmpl::vector< 
        ElectronMethods<>,
        #if(ENABLE_RADIATION == 1)
        Momentum_mt1<>,
        #if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
        RadiationFlag<>,
        #endif
        #endif
        ParticlesData<> 
    >::type
> PIC_Electrons;

}
#endif	/* SPECIES_HPP */

