/**
 * Copyright 2013 Heiko Burau, René Widera
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
 


#ifndef PARTICLESDATA_HPP
#define	PARTICLESDATA_HPP

#include "particles/memory/boxes/TileDataBox.hpp"
#include "types.h"
#include "math/Vector.hpp"
#include "particles/memory/frames/NullFrame.hpp"

namespace picongpu
{
using namespace PMacc;

template<class Base = NullFrame>
class ParticlesData : public Base
{
public:
    typedef float3_X MomType;
    typedef float_X WeightingType;

private:

    PMACC_ALIGN(momentum[Base::tileSize], MomType);

    PMACC_ALIGN(weighting[Base::tileSize], WeightingType);

public:

    template<class OTHERFRAME>
    HDINLINE void copy(lcellId_t myId, OTHERFRAME &other, lcellId_t otherId)
    {
        Base::copy(myId, other, otherId);
        this->getMomentum()[myId] = other.getMomentum()[otherId];
        this->getWeighting()[myId] = other.getWeighting()[otherId];
    }

    HDINLINE VectorDataBox<MomType> getMomentum()
    {
        return VectorDataBox<MomType > (momentum);
    }

    HDINLINE VectorDataBox<WeightingType> getWeighting()
    {
        return VectorDataBox<WeightingType > (weighting);
    }

};

}//namespace picongpu

#endif	/* PARTICLESDATA_HPP */