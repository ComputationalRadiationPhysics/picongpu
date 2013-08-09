/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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
 


#ifndef CURRENTNGP_HPP
#define	CURRENTNGP_HPP

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "dimensions/TVec.h"

namespace picongpu
{
    namespace currentSolverNgp
    {
        using namespace PMacc;

        struct NgpSolver
        {
            typedef TVec < 0, 0, 0 > OffsetOrigin;
            typedef TVec < 1, 1, 1 > OffsetEnd;

            template<class BoxJ, typename PosType, typename VelType, typename ChargeType >
            DINLINE void operator()(BoxJ& boxJ_par, /*box which is shifted to particles cell*/
                                       const PosType pos,
                                       const VelType velocity,
                                       const ChargeType charge, const float3_X& cellSize, const float_X deltaTime)
            {
                typename BoxJ::ValueType j = velocity * charge * deltaTime;
                j.x() *= (1.0 / (cellSize.y() * cellSize.z()));
                j.y() *= (1.0 / (cellSize.x() * cellSize.z()));
                j.z() *= (1.0 / (cellSize.x() * cellSize.y()));

                const DataSpace<DIM3> nearestCell(
                                                  __float2_Xint_rd(pos.x() + float_X(0.5)),
                                                  __float2_Xint_rd(pos.y() + float_X(0.5)),
                                                  __float2_Xint_rd(pos.z() + float_X(0.5))
                                                  );

                atomicAddWrapper(
                                 &(boxJ_par(DataSpace < DIM3 > (0, nearestCell.y(), nearestCell.z())).x()),
                                 j.x());
                atomicAddWrapper(
                                 &(boxJ_par(DataSpace < DIM3 > (nearestCell.x(), 0, nearestCell.z())).y()),
                                 j.y());
                atomicAddWrapper(
                                 &(boxJ_par(DataSpace < DIM3 > (nearestCell.x(), nearestCell.y(), 0)).z()),
                                 j.z());
            }


        };

    }
}

#endif	/* CURRENTNGP_HPP */

