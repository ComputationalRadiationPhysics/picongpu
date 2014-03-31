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

#define DIFF_ADDON 1

#if DIFF_ADDON == 1
    #include "DiffEvolution.hpp"
#endif

namespace gol
{

class Simulation
{
private:
    //TVec<16,16> is arbitrarily chosen SuperCellSize!
    typedef MappingDescription<DIM2, TVec < 16, 16 > > MappingDesc;

    /*DataSpace<DIM2>*/Space gridSize;
    //holds rule mask derived from 23/3 input, see Evolution.hpp
    Evolution<MappingDesc> evo;
    GatherSlice gather;
    
    //typedef GridBuffer<uint8_t, DIM2> Buffer;
    //for storing black and white(live or dead) data for gol
    Buffer* buff1;  //Buffer(see types.h) for swapping between old and new world
    Buffer* buff2;  //like: evolve(buff2 &, const buff1) would work internally
    uint32_t steps;

    bool isMaster;

    #if DIFF_ADDON == 1
        Buffer* bufdif;      //stores what changed from last timestep to next
        MappingDesc mappingdiff;
    #endif

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
        #if DIFF_ADDON == 1
            __delete(bufdif);
        #endif
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
        
    /* getDataSpace will return DataSpace( grid.x +16+16, grid.y +16+16)      */
        evo.init(MappingDesc(layout.getDataSpace(), 1, 1));
    /* Following lines do the same for the Addon as the above line does for   *
     * the Core. mappingdiff stores the layout regarding Core, Border and     *
     * guard in units of supercells to be used by the kernel to identify      *
     * itself. ??? Really don't understand why a new datatype is necessary    *
     * for this, when there is already SubGrid available ???                  */
        #if DIFF_ADDON == 1
            mappingdiff = MappingDesc( layout.getDataSpace(), 1, 1 );
        #endif
        
    /* TODO: I think CommunicationTag was misused here. Second argument is    *
     * bool and signals, whether the size of the buffer is also stored on     *
     * device! Not sure why size could be needed on device ???                *
     *     Before: buff2 = new Buffer(layout, BUFF2);                         */
        buff1 = new Buffer(layout, true);
        buff2 = new Buffer(layout, false);
        #if DIFF_ADDON == 1
            bufdif = new Buffer(layout, true);
        #endif

        /*DataSpace<DIM2>*/ Space guardingCells(1, 1);
    /* TODO: Here here the directions are actually like one would imagine:    *
     * bit 1 to 9 are 0 or 1. (Note that bit 0 is forgotten/unused ??? )      *
     * 1 to 9 represent: left, right, bottom, top, lefttop, leftbottom, ...   *
     * It's not clear which number corresponds to which direction, but also   *
     * doesn't matter here. In 3D this would be 26 directions. In 2D it would *
     * be 8. I don't know why 9 directions are initialized ... ???            */
        for (uint32_t i = 1; i <= 9; ++i)
        {
        /* types.hpp: enum CommunicationTags{ BUFF1 = 0u, BUFF2 = 1u };       */
            buff1->addExchange(GUARD, Mask(i), guardingCells, BUFF1);
            buff2->addExchange(GUARD, Mask(i), guardingCells, BUFF2);
        /* The addon doesn't need this line, because neighbors don't matter   */
            //bufdif->addExchange(GUARD, Mask(i), guardingCells, 3);
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
        evo.initEvolution( buff1->getDeviceBuffer().getDataBox(), 0.1 );

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
        if (isMaster) {
            png(currentStep, picture, gridSize);
        }
        
        #if DIFF_ADDON == 1
            /*********************************************
             * Calculate Difference of those two Buffers *
             *********************************************/
            AreaMapping<CORE,MappingDesc> mapper(mappingdiff);
            __cudaKernel( gol::kernel::diffEvolution )
                        ( mapper.getGridDim(), 
                          MappingDesc::SuperCellSize::getDataSpace() )
                        ( buff1->getDeviceBuffer().getDataBox(),
                          buff2->getDeviceBuffer().getDataBox(),
                          bufdif->getDeviceBuffer().getDataBox(),
                          mapper);
            /* you can comment this out to see exactly what the Borders are.  *
             * They will appear as black spaces/margins around the cores      */
            AreaMapping<BORDER,MappingDesc> mapper2(mappingdiff);
            __cudaKernel( gol::kernel::diffEvolution )
                        ( mapper2.getGridDim(), 
                          MappingDesc::SuperCellSize::getDataSpace() )
                        ( buff1->getDeviceBuffer().getDataBox(),
                          buff2->getDeviceBuffer().getDataBox(),
                          bufdif->getDeviceBuffer().getDataBox(),
                          mapper2);
            bufdif->deviceToHost();
            PMACC_AUTO( difpic, gather( bufdif->getHostBuffer().getDataBox() ) );
            PngCreator pngdiff;
            /* watch out for race conditions when writing to the same file!   *
             * output here is difference between buffer before and thereafter */
            if (isMaster) { 
                pngdiff(currentStep, difpic, gridSize, "dif_");
            }
            
            /* Every MPI-Thread = every CUDA kernel writes it's buffer into a *
             * png file. This also includes the guards.                       *
             * Why do the Guards appear pitch black instead of being copies   *
             * of their neighbor's borders ???                                */
            GridController<DIM2> & gc = Environment<DIM2>::get().GridController();
            PMACC_AUTO(simBox, Environment<DIM2>::get().SubGrid().getSimulationBox());
            PngCreator pngpartial;
            std::stringstream filename;
            filename 
               << "partial_Rank-" << std::setw(3) << std::setfill('0') << gc.getGlobalRank()
               << "_x-Pos-" << std::setw(3) << std::setfill('0') << simBox.getGlobalOffset()[0]
               << "_y-Pos-" << std::setw(3) << std::setfill('0') << simBox.getGlobalOffset()[1] << "\0";
            pngpartial( currentStep, write->getHostBuffer().getDataBox(), write->getGridLayout().getDataSpace(), filename.str() );
        #endif
    }

};
}

#endif	/* SIMULATION_HPP */

