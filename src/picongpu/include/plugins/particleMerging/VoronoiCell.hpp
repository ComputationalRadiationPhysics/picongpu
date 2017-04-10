/* Copyright 2017 Heiko Burau
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

#include "algorithms/KinEnergy.hpp"
#include "pmacc_types.hpp"

namespace picongpu
{
namespace plugins
{
namespace particleMerging
{

    /* Status of a Voronoi cell
     *
     * collecting:
     *  - a Voronoi cell is collecting particles (first state)
     * splitting:
     *  - the Voronoi cell is splitting thus all its particles have
     *    to move to one of two sub-Voronoi cells
     * abort:
     *  - the cell needs to be destroyed. Before this can happen
     *    all its particles need to clear their voronoiCellId attribute.
     * readyForMerging:
     *  - the Voronoi cell is ready for merging. After merging it is destroyed.
     */
    enum struct VoronoiStatus : uint8_t
    {
        collecting,
        splitting,
        abort,
        readyForMerging,
    };


    /* Stage of a Voronoi cell
     *
     * The spliiting process is two-fold: at first, the splitting is done regarding
     * only the spread in position and then by looking at the spread of momentum.
     */
    enum struct VoronoiSplittingStage : bool
    {
        position,
        momentum
    };


    /* Represents a Voronoi cell */
    struct VoronoiCell
    {
        VoronoiStatus status;
        VoronoiSplittingStage splittingStage;
        uint32_t numMacroParticles;
        float_X numRealParticles;

        float3_X meanValue;
        float3_X meanSquaredValue;

        uint8_t splittingComponent;
        int32_t lowerCellId;
        int32_t higherCellId;
        int firstParticleFlag;

        HDINLINE
        VoronoiCell( VoronoiSplittingStage splittingStage = VoronoiSplittingStage::position ) :
            status( VoronoiStatus::collecting ),
            splittingStage( splittingStage ),
            numMacroParticles( 0 ),
            numRealParticles( float_X( 0.0 ) ),
            meanValue( float3_X::create( 0.0 ) ),
            meanSquaredValue( float3_X::create( 0.0 ) ),
            firstParticleFlag( 0 )
        {}


        HDINLINE
        void setToAbort()
        {
            this->status = VoronoiStatus::abort;
        }


        HDINLINE
        void setToSplitting(
            const uint8_t splittingComponent,
            const int32_t lowerCellId,
            const int32_t higherCellId)
        {
            this->status = VoronoiStatus::splitting;
            this->splittingComponent = splittingComponent;
            this->lowerCellId = lowerCellId;
            this->higherCellId = higherCellId;
        }


        HDINLINE
        void setToReadyForMerging()
        {
            this->status = VoronoiStatus::readyForMerging;
        }

        DINLINE
        bool isFirstParticle()
        {
            return atomicExch( &this->firstParticleFlag, 1 ) == 0;
        }


        DINLINE
        void addParticle(
            const floatD_X position,
            const float3_X momentum,
            const float_X weighting
        )
        {
            nvidia::atomicAllInc( &this->numMacroParticles );
            nvidia::atomicAdd( &this->numRealParticles, weighting );

            if( this->splittingStage == VoronoiSplittingStage::position )
            {
                const floatD_X position2 = position * position;

                for( int i = 0; i < simDim; i++ )
                {
                    nvidia::atomicAdd( &this->meanValue[i], weighting * position[i] );
                    nvidia::atomicAdd( &this->meanSquaredValue[i], weighting * position2[i] );
                }
            }
            else
            {
                const float3_X momentum2 = momentum * momentum;

                for( int i = 0; i < DIM3; i++ )
                {
                    nvidia::atomicAdd( &this->meanValue[i], weighting * momentum[i] );
                    nvidia::atomicAdd( &this->meanSquaredValue[i], weighting * momentum2[i] );
                }
            }
        }


        /** finalize mean value calculation */
        HDINLINE
        void finalizeMeanValues()
        {
            this->meanValue /= this->numRealParticles;
            this->meanSquaredValue /= this->numRealParticles;
        }

        HDINLINE
        float_X getMeanEnergy( const float_X mass ) const
        {
            return KinEnergy<>()(
                this->meanValue,
                mass
            );
        }

        HDINLINE
        float_X getMeanMomentum2() const
        {
            return math::abs2( this->meanValue );
        }


        /* determine in which of the two sub-Voronoi cells a particle falls */
        HDINLINE
        int32_t getSubVoronoiCell(
            const floatD_X position,
            const float3_X momentum
        ) const
        {
            const float_X valParticle =
                this->splittingStage == VoronoiSplittingStage::position ?
                position[this->splittingComponent] :
                momentum[this->splittingComponent]
            ;

            const float_X meanVoronoi = this->meanValue[this->splittingComponent];

            return
                valParticle < meanVoronoi ?
                this->lowerCellId :
                this->higherCellId
            ;
        }


        /** calculate the maxmimum squared spread in position
         *
         * @param component index of position component of maxmimum spread
         */
        HDINLINE
        float_X getMaxPositionSpread2( uint8_t& component ) const
        {
            const float3_X meanPosition2 = this->meanValue * this->meanValue;
            const float3_X posSpread2 = this->meanSquaredValue - meanPosition2;

            /* find component of most spread in position */
            component = 0;
            float_X maxPosSpread2 = posSpread2[0];
            for( uint8_t i = 1; i < simDim; i++ )
            {
                if( posSpread2[i] > maxPosSpread2 )
                {
                    maxPosSpread2 = posSpread2[i];
                    component = i;
                }
            }

            return maxPosSpread2;
        }


        /** calculate the maxmimum squared spread in momentum
         *
         * @param component index of momentum component of maxmimum spread
         */
        HDINLINE
        float_X getMaxMomentumSpread2( uint8_t& component ) const
        {
            const float3_X meanMomentum2 = this->meanValue * this->meanValue;
            const float3_X momSpread2 = this->meanSquaredValue - meanMomentum2;

            /* find component of most spread in momentum */
            component = 0;
            float_X maxMomSpread2 = momSpread2[0];
            for( uint8_t i = 1; i < DIM3; i++ )
            {
                if( momSpread2[i] > maxMomSpread2 )
                {
                    maxMomSpread2 = momSpread2[i];
                    component = i;
                }
            }

            return maxMomSpread2;
        }
    };

} // namespace particleMerging
} // namespace plugins
} // namespace picongpu
