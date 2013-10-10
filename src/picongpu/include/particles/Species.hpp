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

#pragma once

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "particles/Particles.hpp"
#include "particles/species/ions/IonMethods.hpp"
#include "particles/species/electrons/ElectronMethods.hpp"

namespace picongpu
{
using namespace PMacc;


/*add old momentum for radiation plugin*/
typedef bmpl::vector<
#if(ENABLE_RADIATION == 1)
    momentum_mt1
#endif
> AttributMomentum_mt1;

/*add old radiation flag for radiation plugin*/
typedef bmpl::vector<
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
    radiationFlag
#endif
> AttributRadiationFlag;

/* not nice, we change this later with nice interfaces*/

typedef
typename JoinVectors<ElectronsDataList, AttributMomentum_mt1, AttributRadiationFlag>::type
Species1_data;

typedef
typename JoinVectors<IonsDataList, AttributMomentum_mt1, AttributRadiationFlag>::type
Species2_data;

typedef Particles<
    Species1_data,
    ElectronsMethodsList
> PIC_Electrons;

typedef Particles<
    Species2_data,
    IonsMethodsList
> PIC_Ions;


/*not nice, but this shuld be changed in the future*/
typedef boost::mpl::vector<
    #if (ENABLE_ELECTRONS == 1)
    PIC_Electrons
    #endif
> Species1;

typedef boost::mpl::vector<
    #if (ENABLE_IONS == 1)
    PIC_Electrons
    #endif
> Species2;

typedef typename JoinVectors<
    Species1,
    Species2
>::type VectorAllSpecies;

} //namespace picongpu
