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
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include "math/vector/Int.hpp"
#include "particles/access/Cell2Particle.hpp"
#include "PhaseSpace.hpp"

#include <fstream>
#include <sstream>

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
    
    template<uint32_t r_dir, uint32_t p_bins, typename SuperCellSize>
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

            /* cell id in this block */
            const int linearCellIdx = particle[particleAccess::LocalCellIdx()].get();
            const PMacc::math::Int<3> cellIdx(
                linearCellIdx  % SuperCellSize::x::value,
                (linearCellIdx % (SuperCellSize::x::value * SuperCellSize::y::value)) / SuperCellSize::x::value,
                linearCellIdx  / (SuperCellSize::x::value * SuperCellSize::y::value) );

            const int r_bin         = cellIdx[r_dir];
            /*const float_X weighting = particle[particleAccess::Weight()];*/
            /* float_X charge    = particle[particleAccess::Charge()];
               const float_X particleChargeDensity = charge / ( CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH );
             */

            const float_X rel_bin = (mom - axis_p_range.first) / (axis_p_range.second - axis_p_range.first);
            int p_bin = int( rel_bin * float_X(p_bins) );
            if( p_bin < 0 )
                p_bin = 0;
            if( p_bin >= p_bins )
                p_bin = p_bins - 1;

            /** \todo take particle shape into account */
            atomicAddWrapper( &(*curDBufferOriginInBlock( r_bin, p_bin )),
                              1 );
        }
    };
    
    template<typename Species, typename SuperCellSize, uint32_t p_bins, uint32_t r_dir>
    struct FunctorBlock
    {
        typedef void result_type;
        
        typedef typename Species::ParticlesBoxType TParticlesBox;
        
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
        DINLINE void operator()( const PMacc::math::Int<3>& indexBlockOffset )
        {
            const PMacc::math::Int<3> indexInBlock( threadIdx.x, threadIdx.y, threadIdx.z );
            const PMacc::math::Int<3> indexGlobal = indexBlockOffset + indexInBlock;
            
            /* create shared mem */
            const uint32_t blockCellsInDir = SuperCellSize::template at<r_dir>::type::value;
            typedef PMacc::math::CT::Int<blockCellsInDir, p_bins> dBufferSizeInBlock;
            container::CT::SharedBuffer<float_X, dBufferSizeInBlock > dBufferInBlock;
            
            /* init shared mem */
            algorithm::cudaBlock::Foreach<SuperCellSize> forEachThreadInBlock;
            {
                using namespace lambda;
                DECLARE_PLACEHOLDERS();
                forEachThreadInBlock( dBufferInBlock.zone(),
                                      dBufferInBlock.origin(),
                                      _1 = float_X(0.0) );
            }
            __syncthreads();

            FunctorParticle<r_dir, p_bins, SuperCellSize> functorParticle;
            particleAccess::Cell2Particle<SuperCellSize> forEachParticleInCell;
            forEachParticleInCell( /* mandatory params */
                                   particlesBox, indexGlobal, functorParticle,
                                   /* optional params */
                                   dBufferInBlock.origin(),
                                   p_element,
                                   axis_p_range
                                 );
            
            __syncthreads();
            /* add to global dBuffer */
            forEachThreadInBlock( dBufferInBlock.zone(),
                                  curOriginPhaseSpace(indexBlockOffset[r_dir], 0),
                                  dBufferInBlock.origin(),
                                  FunctorAtomicAdd() );
                                  
        }
    };
    
    
    template<class AssignmentFunction, class Species>
    PhaseSpace<AssignmentFunction, Species>::PhaseSpace( std::string _name,
                                                          std::string _prefix ) :
    cellDescription(NULL), name(_name), prefix(_prefix), particles(NULL),
    dBuffer(NULL)
    {
        ModuleConnector::getInstance().registerModule(this);

        this->axis_p_range = std::make_pair( -1., 1. );
        this->axis_element = std::make_pair( self::y, self::px );
        this->notifyPeriod = 1;
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::moduleLoad()
    {
        DataConnector::getInstance().registerObserver(this, this->notifyPeriod);
        
        const uint32_t r_element = this->axis_element.first;
        
        /* CORE + BORDER + GUARD elements for spatial bins */
        this->r_bins = SuperCellSize().vec()[r_element]
                     * this->cellDescription->getGridSuperCells()[r_element];
        
        
        this->dBuffer = new container::DeviceBuffer<float_X, 2>( r_bins, this->p_bins );
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::moduleUnload()
    {
        __delete( this->dBuffer );
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::moduleRegisterHelp(po::options_description& desc)
    {
        /*
        desc.add_options()
            ((prefix + ".period").c_str(),
             po::value<uint32_t > (&notifyPeriod)->default_value(0), "enable analyser [for each n-th step]");
         */
    }

    template<class AssignmentFunction, class Species>
    std::string PhaseSpace<AssignmentFunction, Species>::moduleGetName() const
    {
        return this->name;
    }

    template<class AssignmentFunction, class Species >
    template<uint32_t Direction>
    void PhaseSpace<AssignmentFunction, Species>::calcPhaseSpace( )
    {
        const PMacc::math::Int<3> guardCells = SuperCellSize().vec() * size_t(GUARD_SIZE);
        const PMacc::math::Size_t<3> coreBorderSuperCells( this->cellDescription->getGridSuperCells() - 2*int(GUARD_SIZE) );
        const PMacc::math::Size_t<3> coreBorderCells( coreBorderSuperCells.x() * SuperCellSize().vec().x(),
                                                      coreBorderSuperCells.y() * SuperCellSize().vec().y(),
                                                      coreBorderSuperCells.z() * SuperCellSize().vec().z() );

        /* select CORE + BORDER for all cells
         * CORE + BORDER is contiguous, Heiko calls this a "topological spheric zone"
         */
        std::cout << coreBorderCells << " | " << guardCells << std::endl;
        zone::SphericZone<3> zoneCoreBorder( coreBorderCells, guardCells );

        algorithm::kernel::ForeachBlock<SuperCellSize> forEachSuperCell;
        
        FunctorBlock<Species, SuperCellSize, p_bins, Direction> functorBlock(
            this->particles->getDeviceParticlesBox(), dBuffer->origin(),
            this->axis_element.second, this->axis_p_range );

        forEachSuperCell( /* area to work on */
                          zoneCoreBorder,
                          /* data below - passed to functor operator() */
                          cursor::make_MultiIndexCursor<3>(),
                          functorBlock
                        );
    }

    struct PrintIdx
    {
        typedef void result_type;

        DINLINE void operator()(const PMacc::math::Int<3> i) const
        {
            printf("%d,%d,%d\n", i.x(), i.y(), i.z());
        }
    };
    
    struct SetVal
    {
        typedef void result_type;

        cursor::BufferCursor<float_X, 2> curOriginPhaseSpace;
        float_X val;

        SetVal( cursor::BufferCursor<float_X, 2> cur, float_X v ) :
        curOriginPhaseSpace(cur), val(v)
        {}

        DINLINE void operator()(const PMacc::math::Int<2> idx) const
        {
            *curOriginPhaseSpace(idx) = val;
        }
    };

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::notify( uint32_t currentStep )
    {
        std::cout << "[PhaseSpace] notified!" << std::endl;
        
        /* register particle species observer */
        DataConnector &dc = DataConnector::getInstance( );
        this->particles = &(dc.getData<Species > ((uint32_t) Species::FrameType::CommunicationTag, true));

        std::cout << "[PhaseSpace] reset buffer" << std::endl;
        /* reset device buffer */
        //this->dBuffer->assign( float_X(0.0) );
        {
            using namespace lambda;
            DECLARE_PLACEHOLDERS();
            /*
             * matrix: x - elements in r_dir
             *         y - elements for p_bin - divisor of this->pbins
             */
            typedef typename PMacc::math::CT::Int<1, 1, 1> assignBlock;
            algorithm::kernel::Foreach<assignBlock > forEachAssign;
            SetVal functorSetVal(this->dBuffer->origin(), 0.0);
            forEachAssign( /* area to work on */
                           this->dBuffer->zone(),
                           /* data below - passed to functor operator() */
                           cursor::make_MultiIndexCursor<2>(),
                           /* functor */
                           functorSetVal);
            /*
            std::cout << this->dBuffer->zone().size << " | " << this->dBuffer->zone().offset << std::endl;
            algorithm::kernel::Foreach<PMacc::math::CT::Size_t<1, 1, 1> > forEachAssign;
            forEachAssign( this->dBuffer->zone(),
                           cursor::make_MultiIndexCursor<3>(),
                           PrintIdx());
             */
        }

        std::cout << "[PhaseSpace] calc" << std::endl;
        /* calc local phase space */
        if( this->axis_element.first == self::x )
            calcPhaseSpace<self::x>();
        else if( this->axis_element.first == self::y )
            calcPhaseSpace<self::y>();
        else
            calcPhaseSpace<self::z>();

        std::cout << "[PhaseSpace] transfer to host" << std::endl;
        /* transfer to host */
        container::HostBuffer<float_X, 2> hBuffer( this->dBuffer->size() );
        hBuffer = *this->dBuffer;

        /* reduce-add phase space from other GPUs in range [r;r+dr]x[p0;p1]
         * to "lowest" node in range
         * e.g.: phase space x-py: reduce-add all nodes with same x range in
         *                         spatial y and z direction to node with
         *                         lowest y and z position and same x range
         */


        /* gather the full phase space with range [0;rMax]x[p0;p1]
         * to rank 0
         */

        std::cout << "[PhaseSpace] write to file" << std::endl;
        /* write full phase space from rank 0
         */
        std::ostringstream filename;
        filename << "PhaseSpace_" << currentStep << ".dat";
        std::ofstream file(filename.str().c_str());
        file << hBuffer;
    }
    
    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::setMappingDescription(
        MappingDesc* cellDescription )
    {
        this->cellDescription = cellDescription;
    }
}
