/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
 


#ifndef PARTICLEINITRANDOMPOS_HPP
#define	PARTICLEINITRANDOMPOS_HPP

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace particleInitRandomPos
    {

        class particleInitMethods
        {
        public:

            /** Distributes the initial particles uniformly random within the cell.
             * 
             * @param rng a reference to an initialized, UNIFORM random number generator
             * @param totalNumParsPerCell the total number of particles to init for this cell
             * @param curParticle the number of this particle: [0, totalNumParsPerCell-1]
             * @return float3_X with components between [0.0, 1.0)
             */
            template <class UNIRNG>
            DINLINE floatD_X getPosition( UNIRNG& rng,
                                           const uint32_t totalNumParsPerCell,
                                           const uint32_t curParticle )
            {
                floatD_X result;
                for(uint32_t i=0;i<simDim;++i)
                    result[i]=rng();

                return result;
            }

            /** If the particles to initialize (numParsPerCell) end up with a 
             *  related particle weighting (macroWeighting) below MIN_WEIGHTING,
             *  reduce the number of particles if possible to satisfy this condition.
             * 
             * @param numParsPerCell the intendet number of particles for this cell
             * @param realElPerCell  the number of real electrons in this cell
             * @return macroWeighting the intended weighting per macro particle
             */
            DINLINE float_X reduceParticlesToSatisfyMinWeighting( uint32_t& numParsPerCell,
                                                                   const float_X realElPerCell )
            {
                float_X macroWeighting = float_X(0.0);
                if( numParsPerCell > 0 )
                    macroWeighting = realElPerCell / float_X(numParsPerCell);
                
                while( macroWeighting < MIN_WEIGHTING &&
                       numParsPerCell > 0 )
                {
                    --numParsPerCell;
                    if( numParsPerCell > 0 )
                        macroWeighting = realElPerCell / float_X(numParsPerCell);
                    else
                        macroWeighting = float_X(0.0);
                }
                return macroWeighting;
            }

        protected:
        };

    }
}

#endif	/* PARTICLEINITRANDOMPOS_HPP */



