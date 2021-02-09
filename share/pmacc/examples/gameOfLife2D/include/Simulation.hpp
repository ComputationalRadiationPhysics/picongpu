/* Copyright 2013-2021 Rene Widera, Maximilian Knespel, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "types.hpp"
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include "Evolution.hpp"
#include <pmacc/eventSystem/EventSystem.hpp>

#include "GatherSlice.hpp"
#include <pmacc/traits/NumberOfExchanges.hpp>

#include <string>
#include "PngCreator.hpp"

namespace gol
{
    class Simulation
    {
    private:
        /* math::CT::Int<16,16> is arbitrarily chosen SuperCellSize! */
        typedef MappingDescription<DIM2, math::CT::Int<16, 16>> MappingDesc;
        typedef Evolution<MappingDesc> Evolutiontype;

        Space gridSize;
        /* holds rule mask derived from 23/3 input, \see Evolution.hpp */
        Evolutiontype evo;
        GatherSlice gather;

        /* for storing black (dead) and white (alive) data for gol */
        Buffer* buff1; /* Buffer(\see types.h) for swapping between old and new world */
        Buffer* buff2; /* like evolve(buff2 &, const buff1) would work internally */
        uint32_t steps;

        bool isMaster;

    public:
        Simulation(uint32_t rule, int32_t steps, Space gridSize, Space devices, Space periodic)
            : evo(rule)
            , steps(steps)
            , gridSize(gridSize)
            , isMaster(false)
            , buff1(nullptr)
            , buff2(nullptr)
        {
            /* -First this initializes the GridController with number of 'devices'*
             *  and 'periodic'ity. The init-routine will then create and manage   *
             *  the MPI processes and communication group and topology.           *
             * -Second the cudaDevices will be allocated to the corresponding     *
             *  Host MPI processes where hostRank == deviceNumber, if the device  *
             *  is not marked to be used exclusively by another process. This     *
             *  affects: cudaMalloc,cudaKernelLaunch,                             *
             * -Then the CUDA Stream Controller is activated and one stream is    *
             *  added. It's basically a List of cudaStreams. Used to parallelize  *
             *  Memory transfers and calculations.                                *
             * -Initialize TransactionManager                                     */
            Environment<DIM2>::get().initDevices(devices, periodic);

            /* Now we have allocated every node to a grid position in the GC. We  *
             * use that grid position to allocate every node to a position in the *
             * physic grid. Using the localGridSize = the number of cells per     *
             * node = number of cells / nodes, we can get the position of the     *
             * current node as an offset in numbers of cells                      */
            GridController<DIM2>& gc = Environment<DIM2>::get().GridController();
            Space localGridSize(gridSize / devices);

            /* - This forwards arguments to SubGrid.init()                        *
             * - Create Singletons: EnvironmentController, DataConnector,         *
             *                      PluginConnector, nvidia::memory::MemoryInfo   */
            Environment<DIM2>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);
        }

        virtual ~Simulation()
        {
        }

        void finalize()
        {
            gather.finalize();
            __delete(buff1);
            __delete(buff2);
        }

        void init()
        {
            /* subGrid holds global and
             * local SimulationSize and where the local SimArea is in the greater
             * scheme using Offsets from global LEFT, TOP, FRONT
             */
            const SubGrid<DIM2>& subGrid = Environment<DIM2>::get().SubGrid();

            /* The following sets up the local layout which consists of the actual
             * grid cells and some surrounding cells, called guards.
             *
             * ASCII Visualization: example taken for 1D,
             * distributed over 2 GPUs, only 1 border shown between those two GPUs
             * assuming non-periodic boundary conditions.
             * In a N-GPU or periodic example, border cells guard cells exist in each direction.
             * _______GPU 0________       _______GPU 1________
             * | 0 | 1 | 2 | 3 | 4 |      | 3 | 4 | 5 | 6 | 7 |  <-- Global (super)cell idx
             * |___|___|___|___|___|      |___|___|___|___|___|
             * |___Core____|Bor|Gua|      |Gua|Bor|___Core____|
             * |___________|der|rd_|      |rd_|der|___________|
             * |__"real" cells_|***|      |***|__"real" cells_|
             *
             * |***| Clones cells which correspond to the border cells of the neighbor GPU
             *       (sometimes also called "ghost" or "halo" cells/region)
             *
             * Recall that the following is defined:
             *     typedef MappingDescription<DIM2, math::CT::Int<16,16> > MappingDesc;
             * where math::CT::Int<16,16> is arbitrarily(!) chosen SuperCellSize
             * and DIM2 is the dimension of the grid.
             * Expression of 2nd argument translates to DataSpace<DIM3>(16,16,0).
             * This is the guard size (here set to be one Supercell wide in all
             * directions). Meaning we have 16*16*(2*grid.x+2*grid.y+4) more
             * cells in GridLayout than in the SubGrid.
             * The formula above is SuperCellSize * TotalNumGuardCells with (in this case)
             * SuperCellSize = 16*16 (16 cells in 2 dimensions)
             * TotalNumGuardCells =   2 * grid.x (top and bottom)
             *                      + 2 * grid.y (left and right)
             *                      + 4          (the corners)
             */
            GridLayout<DIM2> layout(subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT());

            /* getDataSpace will return DataSpace( grid.x +16+16, grid.y +16+16)  *
             * MappingDesc stores the layout regarding Core, Border and Guard     *
             * in units of SuperCells.                                            *
             * This is saved by init to be used by the kernel to identify itself. */
            evo.init(layout.getDataSpace(), Space::create(1));

            buff1 = new Buffer(layout, false);
            buff2 = new Buffer(layout, false);

            /* Set up the future data exchange. In this case we need to copy the
             * border cells of our neighbors to our guard cells, since we only read
             * from the guard cells but never write to it.
             * guardingCells holds the number of guard(super)cells in each dimension
             */
            Space guardingCells(1, 1);
            for(uint32_t i = 1; i < traits::NumberOfExchanges<DIM2>::value; ++i)
            {
                /* to check which number corresponds to which direction, you can  *
                 * use the following member of class Mask like done in the two    *
                 * lines below:                                                   *
                 * DataSpace<DIM2>relVec = Mask::getRelativeDirections<DIM2>(i);  *
                 * std::cout << "Direction:" << i << " => Vec: (" << relVec[0]    *
                 *           << "," << relVec[1] << ")\n";                        *
                 * The result is: 1:right(1,0), 2:left(-1,0), 3:up(0,1),          *
                 *    4:up right(1,1), 5:(-1,1), 6:(0,-1), 7:(1,-1), 8:(-1,-1)    */

                /* types.hpp: enum CommunicationTags{ BUFF1 = 0u, BUFF2 = 1u };   */
                buff1->addExchange(GUARD, Mask(i), guardingCells, BUFF1);
                buff2->addExchange(GUARD, Mask(i), guardingCells, BUFF2);
            }

            /* Both next lines are defined in GatherSlice.hpp:                   *
             *  -gather saves the MessageHeader object                           *
             *  -Then do an Allgather for the gloabalRanks from GC, sort out     *
             *  -inactive processes (second/boolean ,argument in gather.init) and*
             *   save new MPI_COMMUNICATOR created from these into private var.  *
             *  -return if rank == 0                                             */
            MessageHeader header(gridSize, layout, subGrid.getLocalDomain().offset);
            isMaster = gather.init(header, true);

            /* Calls kernel to initialize random generator. Game of Life is then  *
             * initialized using uniform random numbers. With 10% (second arg)    *
             * white points. World will be written to buffer in first argument    */
            evo.initEvolution(buff1->getDeviceBuffer().getDataBox(), 0.1);
        }

        void start()
        {
            Buffer* read = buff1;
            Buffer* write = buff2;
            for(uint32_t i = 0; i < steps; ++i)
            {
                oneStep(i, read, write);
                std::swap(read, write);
            }
        }

    private:
        void oneStep(uint32_t currentStep, Buffer* read, Buffer* write)
        {
            auto splitEvent = __getTransactionEvent();
            /* GridBuffer 'read' will use 'splitEvent' to schedule transaction    *
             * tasks from the Borders of the neighboring areas to the Guards of   *
             * this local Area added by 'addExchange'. All transactions in        *
             * Transaction Manager will then be done in parallel to the           *
             * calculations in the core. In order to synchronize the data         *
             * transfer for the case the core calculation is finished earlier,    *
             * GridBuffer.asyncComm returns a transaction handle we can check     */
            auto send = read->asyncCommunication(splitEvent);
            evo.run<CORE>(read->getDeviceBuffer().getDataBox(), write->getDeviceBuffer().getDataBox());
            /* Join communication with worker tasks, Now all next tasks run sequential */
            __setTransactionEvent(send);
            /* Calculate Borders */
            evo.run<BORDER>(read->getDeviceBuffer().getDataBox(), write->getDeviceBuffer().getDataBox());
            write->deviceToHost();

            /* gather::operator() gathers all the buffers and assembles those to  *
             * a complete picture discarding the guards.                          */
            auto picture = gather(write->getHostBuffer().getDataBox());
            PngCreator png;
            if(isMaster)
                png(currentStep, picture, gridSize);
        }
    };
} // namespace gol
