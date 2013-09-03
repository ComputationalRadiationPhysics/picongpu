/**
 * Copyright 2013 Axel Huebl, Heiko Burau
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

#include "cuSTL/container/DeviceBuffer.hpp"
#include "math/vector/Int.hpp"
#include "particles/access/Cell2Particle.hpp"

namespace picongpu
{
    using namespace PMacc;
    
    struct FunctorAtomicAdd
    {
        typedef void result_type;
        
        DINLINE void operator()( float_X& dest, const float_X src )
        {
            atomicAddWrapper( &dest, src );
        }
    };
    
    template<uint32_t Direction, uint32_t p_bins>
    struct FunctorParticle
    {
        typedef void result_type;
        
        template<typename Particle, typename Cur >
        DINLINE void operator()( Particle particle,
                                 Cur curDBufferOriginInBlock,
                                 const uint32_t el_p,
                                 const std::pair<float_X, float_X>& axis_p_range )
        {
            const float_X mom       = particle[particleAccess::Mom()].get()[el_p];
            const int r_bin         = particle[particleAccess::LocalCellIdx()].get()[Direction];
            const float_X weighting = particle[particleAccess::Weight()];
            /* float_X charge    = particle[particleAccess::Charge()];
               const float_X particleChargeDensity = charge / ( CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH );
             */
            
            const float_X rel_bin = (mom - axis_p_range.first) / (axis_p_range.second - axis_p_range.first);
            uint32_t p_bin = uint32_t( rel_bin * float_X(p_bins) );
            
            /** \todo take particle shape into account */
            atomicAddWrapper( &(*curDBufferOriginInBlock( r_bin, p_bin )),
                              1 );
        }
    };
    
    template<typename TParticlesBox, typename SuperCellSize, uint32_t p_bins, uint32_t Direction>
    struct FunctorBlock
    {
        typedef void result_type;
        
        TParticlesBox particlesBox;
        cursor::BufferCursor<float_X, 2> curOriginPhaseSpace;
        uint32_t p_element;
        std::pair<float_X, float_X> axis_p_range;
        
        FunctorBlock( const TParticlesBox& pb,
                      cursor::BufferCursor<float_X, 2> cur,
                      const uint32_t p_el,
                      const std::pair<float_X, float_X>& a_ran ) :
        particlesBox(pb), curOriginPhaseSpace(cur), p_element(p_el),
        axis_p_range(a_ran)
        {}
        
        /** Called for the first cell of each block #-of-cells-in-block times
         */
        DINLINE void operator()( const PMacc::math::Int<2>& indexBlockOffset )
        {
            const math::Int<3> indexInBlock( threadIdx.x, threadIdx.y, threadIdx.z );
            const math::Int<3> indexGlobal = indexBlockOffset + indexInBlock;
            
            typedef PMacc::CT::Int<SuperCellSize::template at<Direction>::type::value, p_bins> dBufferSizeInBlock;
            container::CT::SharedBuffer<float_X, dBufferSizeInBlock > dBufferInBlock;
            
            // init shared mem
            using namespace lambda;
            DECLARE_PLACEHOLDERS();
            algorithm::cudaBlock::Foreach<SuperCellSize> forEachThreadInBlock;
            forEachThreadInBlock( dBufferInBlock.zone(),
                                  dBufferInBlock.origin(),
                                  _1 = float_X(0.0) );
            __syncthreads();

            FunctorParticle<Direction, p_bins> functorParticle;
            particleAccess::Cell2Particle<SuperCellSize> forEachParticleInCell;
            forEachParticleInCell( // mandatory params
                                   particlesBox, indexGlobal, functorParticle,
                                   // optional params
                                   dBufferInBlock.origin(),
                                   el_p,
                                   axis_p_range
                                 );
            
            __syncthreads();
            // add to global dBuffer
            forEachThreadInBlock( dBufferInBlock.zone(),
                                  curOriginPhaseSpace(indexBlockOffset),
                                  dBufferInBlock.origin(),
                                  FunctorAtomicAdd() );
                                  
        }
    };

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::moduleLoad()
    {
        DataConnector::getInstance().registerObserver(this, this->notifyFrequency);
        this->particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));
        
        const uint32_t r_element = this->axis_element.first;
        
        // CORE + BORDER + GUARD elements for spatial bins
        this->r_bins = SuperCellSize().vec()[r_element]
                              * this->cellDescription->getGridSuperCells()[r_element];
        
        
        this->dBuffer = new container::DeviceBuffer<float_X, 2>( r_bins, this->p_bins );
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::moduleUnload()
    {
        __delete( this->dBuffer );
    }

    template<class AssignmentFunction, class Species >
    template<uint32_t Direction>
    void PhaseSpace<AssignmentFunction, Species>::calcPhaseSpace( )
    {
        const math::Int<3> guardCells = GUARD_SIZE * SuperCellSize;
        
        // select CORE + BORDER for all cells
        // CORE + BORDER is contigous, Heiko calls this a topological spheric zone
        zone::SphericZone<3> zoneCoreBorder( this->cellDescription->getGridSuperCells(), guardCells );

        algorithm::kernel::ForeachBlock<SuperCellSize> forEachSuperCell;
        
        FunctorBlock<TParticlesBox, SuperCellSize, p_bins, Direction> functorBlock(
            this->particles->getDeviceParticlesBox(), dBuffer->origin(),
            this->axis_element.second, this->axis_p_range );
        
        
        forEachSuperCell( // area to work on
                          zoneCoreBorder,
                          // data below
                          make_MultiIndexCursor(),
                          functorBlock
                        );
    }
    
    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::notify( uint32_t currentStep )
    {
        this->dBuffer->assign(float_X(0.0) );
        
        if( this->axis_element.first() == element_coordinate::x )
            calcPhaseSpace<element_coordinate::x>();
        else if( this->axis_element.first() == element_coordinate::y )
            calcPhaseSpace<element_coordinate::y>();
        else
            calcPhaseSpace<element_coordinate::z>();
    }
}
