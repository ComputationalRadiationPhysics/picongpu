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
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/kernel/ForeachBlock.hpp"
#include "cuSTL/algorithm/mpi/Gather.hpp"
#include "cuSTL/algorithm/mpi/Reduce.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "math/vector/Int.hpp"
#include "math/vector/Size_t.hpp"
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
            const float_X mom_i       = particle[particleAccess::Mom()].get()[el_p];

            /* cell id in this block */
            const int linearCellIdx = particle[particleAccess::LocalCellIdx()].get();
            const PMacc::math::Int<3> cellIdx(
                linearCellIdx  % SuperCellSize::x::value,
                (linearCellIdx % (SuperCellSize::x::value * SuperCellSize::y::value)) / SuperCellSize::x::value,
                linearCellIdx  / (SuperCellSize::x::value * SuperCellSize::y::value) );

            const int r_bin         = cellIdx[r_dir];
            /*const float_X weighting = particle[particleAccess::Weight()].get();*/
            /* float_X charge    = particle[particleAccess::Charge()].get();
               const float_X particleChargeDensity = charge / ( CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH );
             */

            const float_X rel_bin = (mom_i - axis_p_range.first) / (axis_p_range.second - axis_p_range.first);
            int p_bin = int( rel_bin * float_X(p_bins) );

            /* out-of-range bins back to min/max
             * p_bin < 0 ? p_bin = 0;
             * p_bin > (p_bins-1) ? p_bin = p_bins-1;
             */
            p_bin *= int(p_bin >= 0);
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
            forEachThreadInBlock( /* area to work on */
                                  dBufferInBlock.zone(),
                                  /* data below - cursors will be shifted and
                                   * dereferenced */
                                  curOriginPhaseSpace(indexBlockOffset[r_dir], 0),
                                  dBufferInBlock.origin(),
                                  /* functor */
                                  FunctorAtomicAdd() );
                                  
        }
    };
    
    
    template<class AssignmentFunction, class Species>
    PhaseSpace<AssignmentFunction, Species>::PhaseSpace( const std::string _name,
                                                         const std::string _prefix,
                                                         const uint32_t _notifyPeriod,
                                                         const std::pair<float_X, float_X>& _p_range,
                                                         const std::pair<uint32_t, uint32_t>& _element ) :
    cellDescription(NULL), name(_name), prefix(_prefix), particles(NULL),
    dBuffer(NULL), axis_p_range(_p_range), axis_element(_element),
    notifyPeriod(_notifyPeriod)
    {
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

    template<class AssignmentFunction, class Species >
    template<uint32_t r_dir>
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
        
        FunctorBlock<Species, SuperCellSize, p_bins, r_dir> functorBlock(
            this->particles->getDeviceParticlesBox(), dBuffer->origin(),
            this->axis_element.second, this->axis_p_range );

        forEachSuperCell( /* area to work on */
                          zoneCoreBorder,
                          /* data below - passed to functor operator() */
                          cursor::make_MultiIndexCursor<3>(),
                          functorBlock
                        );
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::notify( uint32_t currentStep )
    {
        std::cout << "[PhaseSpace] notified!" << std::endl;
        
        /* register particle species observer */
        DataConnector &dc = DataConnector::getInstance( );
        this->particles = &(dc.getData<Species > ((uint32_t) Species::FrameType::CommunicationTag, true));

        std::cout << "[PhaseSpace] reset buffer" << std::endl;
        /* reset device buffer
         * this->dBuffer->assign( float_X(0.0) );
         */
        using namespace lambda;
        typedef typename PMacc::math::CT::Int<1, 1, 1> assignBlock;
        algorithm::kernel::Foreach<assignBlock > forEachAssign;

        forEachAssign( /* area to work on */
                       this->dBuffer->zone(),
                       /* data below - passed to functor operator() */
                       this->dBuffer->origin(),
                       /* functor */
                       _1 = float_X(0.0));

        std::cout << "[PhaseSpace] calc" << std::endl;
        /* calc local phase space */
        if( this->axis_element.first == This::x )
            calcPhaseSpace<This::x>();
        else if( this->axis_element.first == This::y )
            calcPhaseSpace<This::y>();
        else
            calcPhaseSpace<This::z>();

        std::cout << "[PhaseSpace] transfer to host" << std::endl;
        /* transfer to host */
        container::HostBuffer<float_X, 2> hBuffer( this->dBuffer->size() );
        hBuffer = *this->dBuffer;

        std::cout << "[PhaseSpace] Reduce plane" << std::endl;
        /* reduce-add phase space from other GPUs in range [r;r+dr]x[p0;p1]
         * to "lowest" node in range
         * e.g.: phase space x-py: reduce-add all nodes with same x range in
         *                         spatial y and z direction to node with
         *                         lowest y and z position and same x range
         */
        PMacc::GridController<simDim>& gc = PMacc::GridController<simDim>::getInstance();
        PMacc::math::Size_t<simDim> gpuDim = gc.getGpuNodes();
        PMacc::math::Int<simDim> gpuPos = gc.getPosition();

        /* my plane means: the r_element I am calculating should be 1GPU in width */
        PMacc::math::Size_t<simDim> transversalPlane(gpuDim);
        transversalPlane[this->axis_element.first] = 1;
        /* my plane means: the offset for the transversal plane to my r_element
         * should be zero
         */
        PMacc::math::Int<simDim> longOffset(0);
        longOffset[this->axis_element.first] = gpuPos[this->axis_element.first];

        zone::SphericZone<simDim> zoneTransversalPlane( transversalPlane, longOffset );

        /* Am I the lowest GPU in my plane? */
        PMacc::math::Int<simDim> planePos(gpuPos);
        planePos[this->axis_element.first] = 0;
        const bool isLowestGPUinPlane = ( planePos == PMacc::math::Int<simDim>(0) );
        
        algorithm::mpi::Reduce<simDim> planeReduce( zoneTransversalPlane, isLowestGPUinPlane );
        container::HostBuffer<float_X, 2> hReducedBuffer( hBuffer.size() );
        planeReduce( /* dst, src */
                     hReducedBuffer,
                     hBuffer,
                     /* the functors return value will be written to dst */
                     _1 + _2 );

        /** all non-reduce-root processes are done now */
        if( !isLowestGPUinPlane )
            return;

        std::cout << "[PhaseSpace] Communicate and add GUARDS (todo)" << std::endl;
        /** \todo communicate GUARD and add it to the two neighbors BORDER */
        PMacc::SubGrid<simDim>& sg = PMacc::SubGrid<simDim>::getInstance();
        container::HostBuffer<float_X, 2> hReducedBuffer_noGuard( sg.getSimulationBox().getLocalSize()[this->axis_element.first],
                                                                  this->p_bins );
        algorithm::host::Foreach forEachCopyWithoutGuard;
        forEachCopyWithoutGuard(/* area to work on */
                                hReducedBuffer_noGuard.zone(),
                                /* data below - passed to functor operator() */
                                hReducedBuffer.origin()(SuperCellSize().vec()[this->axis_element.first] * GUARD_SIZE, 0),
                                hReducedBuffer_noGuard.origin(),
                                /* functor */
                                _2 = _1);

        std::cout << "[PhaseSpace] write to file" << std::endl;
        /* write full phase space from rank 0
         */
        std::string fCoords("xyz");
        
        std::ostringstream filename;
        filename << "PhaseSpace_"
                 << currentStep
                 /* local area in the spatial range */
                 << "_" << gpuPos[this->axis_element.first]
                 /* _xpx or _ypz or ... */
                 << "_" << fCoords.at(this->axis_element.first)
                 << "p" << fCoords.at(this->axis_element.second)
                 << ".dat";
        std::ofstream file(filename.str().c_str());
        file << hReducedBuffer_noGuard;
    }
    
    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::setMappingDescription(
        MappingDesc* cellDescription )
    {
        this->cellDescription = cellDescription;
    }
}
