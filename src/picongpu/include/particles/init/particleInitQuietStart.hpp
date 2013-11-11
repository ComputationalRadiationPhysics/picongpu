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
 


#ifndef PARTICLEINITQUIETSTART_HPP
#define	PARTICLEINITQUIETSTART_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "ppFunctions.hpp"

namespace picongpu
{
    namespace particleInitQuietStart
    {

        class particleInitMethods
        {
        public:

            DINLINE particleInitMethods( ) :
                numParsPerCell_X( NUM_PARTICLES_PER_CELL_X ),
                numParsPerCell_Y( NUM_PARTICLES_PER_CELL_Y ),
                numParsPerCell_Z( NUM_PARTICLES_PER_CELL_Z )
            {

            }

            /** Distributes the initial particles lattice-like within the cell.
             * 
             * @param rng a reference to an initialized, UNIFORM random number generator
             * @param totalNumParsPerCell the total number of particles to init for this cell
             * @param curParticle the number of this particle: [0, totalNumParsPerCell-1]
             * @return float3_X with components between [0.0, 1.0)
             */
            template <class UNIRNG>
            DINLINE float3_X getPosition( UNIRNG& rng,
                                           const uint32_t totalNumParsPerCell,
                                           const uint32_t curParticle )
            {
                // spacing between particles in each direction in the cell
                const float3_X spacing = float3_X( float_X(1.0) / float_X(numParsPerCell_X),
                                                    float_X(1.0) / float_X(numParsPerCell_Y),
                                                    float_X(1.0) / float_X(numParsPerCell_Z) );
                // length of the x lattice, number of particles in the xy plane
                const uint32_t lineX   = numParsPerCell_X;
                const uint32_t planeXY = numParsPerCell_X * numParsPerCell_Y;
                
                // coordinate in the local in-cell lattice
                //   x = [0, numParsPerCell_X-1]
                //   y = [0, numParsPerCell_Y-1]
                //   z = [0, numParsPerCell_Z-1]
                const uint3 inCellCoordinate = make_uint3( curParticle % lineX,
                                                           (curParticle % planeXY) / lineX,
                                                           curParticle / planeXY );
                
                return float3_X( float_X(inCellCoordinate.x) * spacing.x() + spacing.x() * 0.5,
                                    float_X(inCellCoordinate.y) * spacing.y() + spacing.y() * 0.5,
                                    float_X(inCellCoordinate.z) * spacing.z() + spacing.z() * 0.5 );
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
                    //--numParsPerCell;
                    PMACC_MAX_DO( --, numParsPerCell_X, PMACC_MAX( numParsPerCell_Y, numParsPerCell_Z ) );
                    numParsPerCell = numParsPerCell_X * numParsPerCell_Y * numParsPerCell_Z;

                    if( numParsPerCell > 0 )
                        macroWeighting = realElPerCell / float_X(numParsPerCell);
                    else
                        macroWeighting = float_X(0.0);
                }
                return macroWeighting;
            }

        protected:

            uint32_t numParsPerCell_X;
            uint32_t numParsPerCell_Y;
            uint32_t numParsPerCell_Z;
        };
    }
}

#endif	/* PARTICLEINITQUIETSTART_HPP */



