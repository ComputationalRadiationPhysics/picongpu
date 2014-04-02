/**
 * Copyright 2013 Axel Huebl, Rene Widera
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
#include "compileTime/conversion/MakeSeq.hpp"

#include "particles/Particles.hpp"
#include "particles/species/ions/IonMethods.hpp"
#include "particles/species/electrons/ElectronMethods.hpp"
#include "particles/ParticleDescription.hpp"
#include <boost/mpl/string.hpp>

namespace picongpu
{
using namespace PMacc;


/*add old momentum for radiation plugin*/
typedef typename MakeSeq<
#if(ENABLE_RADIATION == 1)
momentumPrev1
#endif
>::type AttributMomentum_mt1;

/*add old radiation flag for radiation plugin*/
typedef typename MakeSeq<
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
radiationFlag
#endif
>::type AttributRadiationFlag;

/** \todo: not nice, we change this later with nice interfaces*/

typedef
typename MakeSeq<
        ElectronsDataList,
        AttributMomentum_mt1,
        AttributRadiationFlag
    >::type
Species1_data;

typedef
typename MakeSeq<
        IonsDataList,
        AttributMomentum_mt1,
        AttributRadiationFlag
    >::type
Species2_data;

typedef Particles<
    ParticleDescription<
        boost::mpl::string<'e'>,
        MappingDesc::SuperCellSize,
        Species1_data,
        ElectronsMethodsList
    >
> PIC_Electrons;

typedef Particles<
    ParticleDescription<
        boost::mpl::string<'i'>,
        MappingDesc::SuperCellSize,
        Species2_data,
        IonsMethodsList
    >
> PIC_Ions;

/** \todo: not nice, but this should be changed in the future*/
typedef typename MakeSeq<
#if (ENABLE_ELECTRONS == 1)
PIC_Electrons
#endif
>::type Species1;

typedef typename MakeSeq<
#if (ENABLE_IONS == 1)
PIC_Ions
#endif
>::type Species2;

typedef typename MakeSeq<
Species1,
Species2
>::type VectorAllSpecies;

} //namespace picongpu

#include "particles/Particles.tpp"
