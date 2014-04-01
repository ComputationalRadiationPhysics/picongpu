/**
 * Copyright 2013-2014 Rene Widera, Maximilian Knespel
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


#ifndef SIMULATION_HPP
#define	SIMULATION_HPP

#include "types.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "Evolution.hpp"
#include "eventSystem/EventSystem.hpp"

#include "GatherSlice.hpp"



#include <string>
#include "PngCreator.hpp"

namespace gol
{

class Simulation
{
private:
    //TVec<16,16> is arbitrarily chosen SuperCellSize!
    typedef MappingDescription<DIM2, TVec < 16, 16 > > MappingDesc;
    typedef Evolution<MappingDesc> Evolutiontype;

    Space/*DataSpace<DIM2>*/ gridSize;
    //holds rule mask derived from 23/3 input, see Evolution.hpp
    Evolutiontype evo;
    GatherSlice gather;

    //typedef GridBuffer<uint8_t, DIM2> Buffer;
    //for storing black and white(live or dead) data for gol
    Buffer* buff1;  //Buffer(see types.h) for swapping between old and new world
    Buffer* buff2;  //like: evolve(buff2 &, const buff1) would work internally
    uint32_t steps;

    bool isMaster;

public:

    Simulation(uint32_t rule, int32_t steps, Space gridSize, Space devices, Space periodic) :
    evo(rule), steps(steps), gridSize(gridSize), isMaster(false), buff1(NULL), buff2(NULL)
    {
    /* - First this initializes the GridController with number of 'devices'   *
     *   and 'periodic'ity. The init-routine will then create and manage the  *
     *   MPI processes and communication group and topology.                  *
     * - Second the cudaDevices will be allocated to the corresponding Host   *
     *   MPI processes where hostRank == deviceNumber, if the device is not   *
     *   marked to be used exclusively by another process. This affects:      *
     *   cudaMalloc,cudaKernelLaunch,                                         *
     * - Then the CUDA Stream Controller is activated and one stream is added * 
     *   It's basically a List of cudaStreams. Used to parallelize Memory     *
     *   transfers and calculations.                                          *
     * - Initialize TransactionManager                                        *
     **************************************************************************/
        Environment<DIM2>::get().initDevices(devices, periodic);
    /* Now we have allocated every node to a grid position in the GC. We use  *
     * that grid position to allocate every node to a position in the physic  *
     * grid. Using the localGridSize = the number of cells per node = number  *
     * of cells / nodes, we can get the position of the current node as an    *
     * offset in numbers of cells                                             */
        GridController<DIM2> & gc = Environment<DIM2>::get().GridController(); 
        Space/*DataSpace<DIM2>*/ localGridSize(gridSize / devices);
    /* - First this forwards its arguments to SubGrid.init(), which saves     *
     *   these inside a private SimulationBox Object                          *
     * - Create Singletons: EnvironmentController, DataConnector,             *
     *                      PluginConnector, nvidia::memory::MemoryInfo       */
        Environment<DIM2>::get().initGrids( gridSize, localGridSize,
                                            gc.getPosition() * localGridSize);
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
    /* copy singleton simulationBox data to simbox (#define would do the      *
     * same) simBox holds global and local SimulationSize and where the local *
     * SimArea is in the greater scheme using Offsets from global LEFT,TOP,   *
     * FRONT                                                                  */
        PMACC_AUTO(simBox, Environment<DIM2>::get().SubGrid().getSimulationBox());
        
    /* Recall that in types.hpp following is defined:                         *
     *     typedef MappingDescription<DIM2, TVec<16,16> > MappingDesc;        *
     * where TVec<16,16> is arbitrarily(!) chosen SuperCellSize and DIM2 is   *
     * the dimension of the grid.                                             *
     * Expression of 2nd argument translates to DataSpace<DIM3>(16,16,0).     *
     * This is the guard size (here set to be one Supercell wide in all       *
     * directions). Meaning we have 16*16*(2*grid.x+2*grid.y+4) more cells in *
     * GridLayout than in SimulationBox.                                      */
        GridLayout<DIM2> layout( simBox.getLocalSize(), 
                                 MappingDesc::SuperCellSize::getDataSpace());
        
    /* getDataSpace will return DataSpace( grid.x +16+16, grid.y +16+16)      *
     * init stores the arguments internally in a MappingDesc private variable *
     * which stores the layout regarding Core, Border and guard in units of   *
     * SuperCells to be used by the kernel to identify itself.                *
     * Don't understand why a new datatype is necessary for this, when there  *
     * is already SubGrid available ???                                       */
        evo.init(MappingDesc(layout.getDataSpace(), 1, 1));

    /* ??? I think CommunicationTag is misused here. Second argument should   *
     * be bool signaling, whether the size of the buffer is also stored on    *
     * device! Not sure why size could be needed on device ???                *
     *     Should be: buff1 = new Buffer(layout, false);                      *
     *                buff2 = new Buffer(layout, true);                       */
        buff1 = new Buffer(layout, BUFF1);
        buff2 = new Buffer(layout, BUFF2);

        Space/*DataSpace<DIM2>*/ gardingCells(1, 1);
    /* TODO: Here here the directions are actually like one would imagine:    *
     * bit 1 to 9 are 0 or 1. (Note that bit 0 is forgotten/unused ??? )      *
     * 1 to 9 represent: left, right, bottom, top, lefttop, leftbottom, ...   *
     * It's not clear which number corresponds to which direction, but also   *
     * doesn't matter here. In 3D this would be 26 directions. In 2D it would *
     * be 8. I don't know why 9 directions are initialized ... ???            */
        for (uint32_t i = 1; i <= 9; ++i)
        {
            //types.hpp: enum CommunicationTags{ BUFF1 = 0u, BUFF2 = 1u };
            buff1->addExchange(GUARD, Mask(i), gardingCells, BUFF1);
            buff2->addExchange(GUARD, Mask(i), gardingCells, BUFF2);
        }
    /* In contrast to this usage of directions there exists an enum in        *
     * libPMacc/include/types.h:                                              *
     *    enum ExchangeType { RIGHT = 1u, LEFT = 2u, BOTTOM = 3u,             *
     *                        TOP   = 6u, BACK = 9u, FRONT  = 18u   };        *
     * Meaning we have said something like addExchange(GUARD, Mask(BACK),...) *
     * But even so, buff1->getSendMask().containsExchangeType(i) returns true *
     * for 0<=i<=8 and false for i=9, which is correct ... ???                */
        
     /* Both next lines are defined in GatherSlice.hpp:                       *
      *  - gather saves the MessageHeader object with all its argument input  *
      *  - Then do an Allgather for the gloabalRanks from GC, sort out        *
      *  - inactive processes (second/boolean ,argument in gather.init) and   *
      *    save new MPI_COMMUNICATOR created from these into private var.     *
      *  - return rank == 0                                                   */
        MessageHeader header(gridSize, layout, simBox.getGlobalOffset());
        isMaster = gather.init(header, true);

    /* Calls kernel to initialize random generator. Game of Life is then      *
     * initialized using uniform random numbers. With 10% (second arg) white  *
     * points. World will be written to buffer in first argument              */
        evo.initEvolution(buff1->getDeviceBuffer().getDataBox(), 0.1);

    }

    void start()
    {
        Buffer* read = buff1;
        Buffer* write = buff2;
        for (uint32_t i = 0; i < steps; ++i)
        {
            oneStep(i, read, write);
            std::swap(read, write);
        }
    }
private:

    void oneStep(uint32_t currentStep, Buffer* read, Buffer* write)
    {
        /* Environment<>::get().TransactionManager().getTransactionEvent() <=>*/
        PMACC_AUTO(splitEvent, __getTransactionEvent());
        /* GridBuffer 'read' will use 'splitEvent' to schedule transaction    *
         * tasks from the Guard of this local Area to the Borders of the      *
         * neighboring areas added by 'addExchange'. All transactions in      *
         * Transaction Manager will then be done in parallel to the           *
         * calculations in the core. In order to synchronize the data         *
         * transfer for the case the core calculation is finished earlier,    *
         * GridBuffer.asyncComm returns a transaction handle we can check     */
        PMACC_AUTO(send, read->asyncCommunication(splitEvent));
        evo.run<CORE>( read->getDeviceBuffer().getDataBox(),
                       write->getDeviceBuffer().getDataBox() );
        /* Join communication with worker tasks, Now all next tasks run sequential  */
        __setTransactionEvent(send);
        /* Calculate Borders */
        evo.run<BORDER>( read->getDeviceBuffer().getDataBox(), 
                         write->getDeviceBuffer().getDataBox() );
        write->deviceToHost();

        /* We need PMACC_AUTO here, because DataBox<base>, base is not easily *
         * known. In this case we know, that base is except one case in       *
         * PiConGPU PitchedBox<Type,Dim>. But we would have to derive these   *
         * two template parameters for PitchedBox from steps waaay before     *
         * this one : DataBox<PitchedBox<?,DIM2>> picture = ...               *
         * gather::operator() gathers all the buffers and assembles those to  *
         * a complete picture discarding the guards.                          */
        PMACC_AUTO(picture, gather(write->getHostBuffer().getDataBox()));
        PngCreator png;
        if (isMaster) png(currentStep, picture, gridSize);

    }

    //Not used anymore, because it's now in Environment.hpp ???
    void setDevice(int deviceNumber)
    {
        int num_gpus = 0; //count of gpus
        cudaGetDeviceCount(&num_gpus);
        //##ERROR handling
        if (num_gpus < 1) //check if cuda device ist found
        {
            throw std::runtime_error("no CUDA capable devices detected");
        }
        else if (num_gpus < deviceNumber) //check if i can select device with diviceNumber
        {
            std::cerr << "no CUDA device " << deviceNumber << ", only " << num_gpus << " devices found" << std::endl;
            throw std::runtime_error("CUDA capable devices can't be selected");
        }

        cudaDeviceProp devProp;
        cudaError rc;
        CUDA_CHECK(cudaGetDeviceProperties(&devProp, deviceNumber));
        if (devProp.computeMode == cudaComputeModeDefault)
        {
            CUDA_CHECK(rc = cudaSetDevice(deviceNumber));
            if (cudaSuccess == rc)
            {
                cudaDeviceProp dprop;
                cudaGetDeviceProperties(&dprop, deviceNumber);
                //!\todo: write this only on debug
                log<ggLog::CUDA_RT > ("Set device to %1%: %2%") % deviceNumber % dprop.name;
            }
        }
        else
        {
            //gpu mode is cudaComputeModeExclusiveProcess and a free device is automaticly selected.
            log<ggLog::CUDA_RT > ("Device is selected by CUDA automaticly. (because cudaComputeModeDefault is not set)");
        }
        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    }

};
}

#endif	/* SIMULATION_HPP */

