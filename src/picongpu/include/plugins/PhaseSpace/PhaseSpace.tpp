/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau
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

#include <vector>
#include <algorithm>

#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/kernel/ForeachBlock.hpp"
#include "cuSTL/algorithm/mpi/Gather.hpp"
#include "cuSTL/algorithm/mpi/Reduce.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "math/vector/Int.hpp"
#include "math/vector/Size_t.hpp"

#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"

#include "PhaseSpace.hpp"
#include "PhaseSpaceFunctors.hpp"

#include "DumpHBufferSplashP.hpp"

namespace picongpu
{
    using namespace PMacc;

    template<class AssignmentFunction, class Species>
    PhaseSpace<AssignmentFunction, Species>::PhaseSpace( const std::string _name,
                                                         const std::string _prefix,
                                                         const uint32_t _notifyPeriod,
                                                         const std::pair<float_X, float_X>& _p_range,
                                                         const std::pair<uint32_t, uint32_t>& _element ) :
    cellDescription(NULL), name(_name), prefix(_prefix), particles(NULL),
    dBuffer(NULL), axis_p_range(_p_range), axis_element(_element),
    notifyPeriod(_notifyPeriod), isPlaneReduceRoot(false),
    commFileWriter(MPI_COMM_NULL), planeReduce(NULL)
    {
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::pluginLoad()
    {
        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

        const uint32_t r_element = this->axis_element.first;

        /* CORE + BORDER + GUARD elements for spatial bins */
        this->r_bins = SuperCellSize().toRT()[r_element]
                     * this->cellDescription->getGridSuperCells()[r_element];

        this->dBuffer = new container::DeviceBuffer<float_PS, 2>( r_bins, this->num_pbins );

        /* reduce-add phase space from other GPUs in range [r;r+dr]x[p0;p1]
         * to "lowest" node in range
         * e.g.: phase space x-py: reduce-add all nodes with same x range in
         *                         spatial y and z direction to node with
         *                         lowest y and z position and same x range
         */
        PMacc::GridController<simDim>& gc = PMacc::Environment<simDim>::get().GridController();
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
        this->isPlaneReduceRoot = ( planePos == PMacc::math::Int<simDim>(0) );

        this->planeReduce = new algorithm::mpi::Reduce<simDim>( zoneTransversalPlane,
                                                                this->isPlaneReduceRoot );

        /* Create communicator with ranks of each plane reduce root */
        {
            /* Array with root ranks of the planeReduce operations */
            std::vector<int> planeReduceRootRanks( gc.getGlobalSize(), -1 );
            /* Am I one of the planeReduce root ranks? my global rank : -1 */
            int myRootRank = gc.getGlobalRank() * this->isPlaneReduceRoot
                           - ( ! this->isPlaneReduceRoot );

            MPI_Group world_group, new_group;
            MPI_CHECK(MPI_Allgather( &myRootRank, 1, MPI_INT,
                                     &(planeReduceRootRanks.front()),
                                     1,
                                     MPI_INT,
                                     MPI_COMM_WORLD ));

            /* remove all non-roots (-1 values) */
            std::sort( planeReduceRootRanks.begin(), planeReduceRootRanks.end() );
            std::vector<int> ranks( std::lower_bound( planeReduceRootRanks.begin(),
                                                      planeReduceRootRanks.end(),
                                                      0 ),
                                    planeReduceRootRanks.end() );

            MPI_CHECK(MPI_Comm_group( MPI_COMM_WORLD, &world_group ));
            MPI_CHECK(MPI_Group_incl( world_group, ranks.size(), ranks.data(), &new_group ));
            MPI_CHECK(MPI_Comm_create( MPI_COMM_WORLD, new_group, &commFileWriter ));
            MPI_CHECK(MPI_Group_free( &new_group ));
            MPI_CHECK(MPI_Group_free( &world_group ));
        }
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::pluginUnload()
    {
        __delete( this->dBuffer );
        __delete( planeReduce );

        if( commFileWriter != MPI_COMM_NULL )
            MPI_CHECK(MPI_Comm_free( &commFileWriter ));
    }

    template<class AssignmentFunction, class Species >
    template<uint32_t r_dir>
    void PhaseSpace<AssignmentFunction, Species>::calcPhaseSpace( )
    {
        const PMacc::math::Int<DIM3> guardCells = precisionCast<int>(SuperCellSize().toRT()) * int(GUARD_SIZE);
        const PMacc::math::Size_t<DIM3> coreBorderSuperCells( this->cellDescription->getGridSuperCells() - 2*int(GUARD_SIZE) );
        const PMacc::math::Size_t<DIM3> coreBorderCells = coreBorderSuperCells *
            precisionCast<size_t>( SuperCellSize().toRT() );

        /* select CORE + BORDER for all cells
         * CORE + BORDER is contiguous, in cuSTL we call this a "topological spheric zone"
         */
        zone::SphericZone<DIM3> zoneCoreBorder( coreBorderCells, guardCells );

        algorithm::kernel::ForeachBlock<SuperCellSize> forEachSuperCell;

        FunctorBlock<Species, SuperCellSize, float_PS, num_pbins, r_dir> functorBlock(
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
        /* register particle species observer */
        DataConnector &dc = Environment<>::get().DataConnector();
        this->particles = &(dc.getData<Species > (Species::FrameType::getName(), true));

        /* reset device buffer */
        this->dBuffer->assign( float_PS(0.0) );

        /* calculate local phase space */
        if( this->axis_element.first == This::x )
            calcPhaseSpace<This::x>();
        else if( this->axis_element.first == This::y )
            calcPhaseSpace<This::y>();
        else
            calcPhaseSpace<This::z>();

        /* transfer to host */
        container::HostBuffer<float_PS, 2> hBuffer( this->dBuffer->size() );
        hBuffer = *this->dBuffer;

        /* reduce-add phase space from other GPUs in range [r;r+dr]x[p0;p1]
         * to "lowest" node in range
         * e.g.: phase space x-py: reduce-add all nodes with same x range in
         *                         spatial y and z direction to node with
         *                         lowest y and z position and same x range
         */
        using namespace lambda;
        container::HostBuffer<float_PS, 2> hReducedBuffer( hBuffer.size() );
        (*this->planeReduce)( /* parameters: dest, source */
                             hReducedBuffer,
                             hBuffer,
                             /* the functors return value will be written to dst */
                             _1 + _2 );

        /** all non-reduce-root processes are done now */
        if( !this->isPlaneReduceRoot )
            return;

        /** \todo communicate GUARD and add it to the two neighbors BORDER */

        /** prepare local output buffer of the phase space*/
        PMacc::SubGrid<simDim>& sg = Environment<simDim>::get().SubGrid();
        container::HostBuffer<float_PS, 2> hReducedBuffer_noGuard( sg.getSimulationBox().getLocalSize()[this->axis_element.first],
                                                                  this->num_pbins );
        algorithm::host::Foreach forEachCopyWithoutGuard;
        forEachCopyWithoutGuard(/* area to work on */
                                hReducedBuffer_noGuard.zone(),
                                /* data below - passed to functor operator() */
                                hReducedBuffer.origin()(SuperCellSize().toRT()[this->axis_element.first] * GUARD_SIZE, 0),
                                hReducedBuffer_noGuard.origin(),
                                /* functor */
                                _2 = _1);

        /* write to file */
        const double UNIT_VOLUME = ( UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH );
        const double unit = UNIT_CHARGE / UNIT_VOLUME;
        DumpHBuffer dumpHBuffer;

        if( this->commFileWriter != MPI_COMM_NULL )
            dumpHBuffer( hReducedBuffer_noGuard, this->axis_element, unit, currentStep, this->commFileWriter );
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::setMappingDescription(
        MappingDesc* cellDescription )
    {
        this->cellDescription = cellDescription;
    }

} /* namespace picongpu */
