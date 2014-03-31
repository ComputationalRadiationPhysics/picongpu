/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#pragma once

#include "types.hpp"
#include "dimensions/TVec.h"
#include "mappings/threads/ThreadCollective.hpp"
#include "nvidia/functors/Assign.hpp"
#include "memory/boxes/CachedBox.hpp"
#include "memory/dataTypes/Mask.hpp"

#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "nvidia/rng/distributions/Uniform_float.hpp"

namespace gol
{
    namespace kernel
    {
        using namespace PMacc;

        template<class BoxReadOnly, class BoxWriteOnly, class Mapping>
        __global__ void diffEvolution(BoxReadOnly buffRead,
                                  BoxWriteOnly buffWrite,
                                  BoxWriteOnly buffDiff,
                                  Mapping mapper)
        {
            typedef typename BoxReadOnly::ValueType Type;
            typedef SuperCellDescription< typename Mapping::SuperCellSize, TVec<1,1>, TVec<1,1> > BlockArea;
            PMACC_AUTO(cache, CachedBox::create < 0, Type > (BlockArea()));

            const Space block(mapper.getSuperCellIndex(Space(blockIdx)));
            const Space blockCell = block * Mapping::SuperCellSize();
            const Space threadIndex(threadIdx);
            PMACC_AUTO(buffRead_shifted, buffRead.shift(blockCell));

            ThreadCollective<BlockArea> collectiv(threadIndex);

            nvidia::functors::Assign assign;
            collectiv( assign, cache, buffRead_shifted );
            __syncthreads();

            /*Type neighbors = 0;
            for (uint32_t i = 1; i < 9; ++i)
            {
                Space offset(Mask::getRelativeDirections<DIM2 > (i));
                neighbors += cache(threadIndex + offset);
            }

            Type isLife = cache(threadIndex);
            isLife = 2;*/
            //isLife = (bool)(((!isLife)*(1 << (neighbors + 9))) & rule) + (bool)(((isLife)*(1 << (neighbors))) & rule);

            buffDiff(blockCell + threadIndex) = (buffRead(blockCell + threadIndex) == buffWrite(blockCell + threadIndex));
        }
    }
}



