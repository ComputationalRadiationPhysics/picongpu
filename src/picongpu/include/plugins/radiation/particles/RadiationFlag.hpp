/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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

#include "particles/memory/boxes/TileDataBox.hpp"
#include "pmacc_types.hpp"
#include "particles/memory/frames/NullFrame.hpp"

namespace picongpu
{
    using namespace PMacc;


    template<class Base = NullFrame>
    class RadiationFlag : public Base
    {
    private:
        PMACC_ALIGN(radiationFlag[Base::tileSize], bool);

    public:

        template<class OTHERFRAME>
        HDINLINE void copy(lcellId_t myId, OTHERFRAME &other, lcellId_t otherId)
        {
            Base::copy(myId, other, otherId);
            this->getRadiationFlag()[myId] = other.getRadiationFlag()[otherId];
        }

        HDINLINE VectorDataBox<bool> getRadiationFlag()
        {
            return VectorDataBox<bool > (radiationFlag);
        }

    };


}//namespace picongpu


