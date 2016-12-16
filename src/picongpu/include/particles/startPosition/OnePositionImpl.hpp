/**
 * Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera
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
#include "particles/startPosition/MacroParticleCfg.hpp"
#include "particles/startPosition/IFunctor.def"

namespace picongpu
{
namespace particles
{
namespace startPosition
{

    template< typename T_ParamClass >
    struct OnePositionImpl
    {

        typedef T_ParamClass ParamClass;

        template< typename T_SpeciesType >
        struct apply
        {
            typedef OnePositionImpl< ParamClass > type;
        };

        HINLINE
        OnePositionImpl( uint32_t )
        {
        }

        DINLINE void
        init( const DataSpace<simDim>& /*totalCellOffset*/ )
        {
        }

        /** Distributes the initial particles all on one position in the cell.
         *
         * @param curParticle the number of this particle: [0, totalNumParsPerCell-1]
         * @return float3_X with components between [0.0, 1.0)
         */
        DINLINE floatD_X
        operator()( const uint32_t curParticle )
        {
            return ParamClass().
                   inCellOffset.
                   template shrink< simDim >();
        }

        /** If the particles to initialize (numParsPerCell) end up with a
         *  related particle weighting (macroWeighting) below MIN_WEIGHTING,
         *  reduce the number of particles if possible to satisfy this condition.
         *
         * @param realParticlesPerCell  the number of real particles in this cell
         * @return macroWeighting the intended weighting per macro particle
         */
        DINLINE MacroParticleCfg
        mapRealToMacroParticle( const float_X realParticlesPerCell )
        {
            uint32_t numParsPerCell = ParamClass::numParticlesPerCell;
            float_X macroWeighting = float_X(0.0);
            if( numParsPerCell > 0 )
                macroWeighting =
                    realParticlesPerCell /
                    float_X( numParsPerCell );

            while(
                macroWeighting < MIN_WEIGHTING &&
                numParsPerCell > 0
            )
            {
                --numParsPerCell;
                if( numParsPerCell > 0 )
                    macroWeighting =
                        realParticlesPerCell /
                        float_X( numParsPerCell );
                else
                    macroWeighting = float_X( 0.0 );
            }
            MacroParticleCfg macroParCfg;
            macroParCfg.weighting = macroWeighting;
            macroParCfg.numParticlesPerCell = numParsPerCell;

            return macroParCfg;
        }
    };
} // namespace startPosition
} // namespace particles
} // namespace picongpu
