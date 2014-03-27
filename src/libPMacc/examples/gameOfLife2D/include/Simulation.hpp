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

    Space gridSize;              //typedef DataSpace<DIM2> Space; (I hate typedefs ...)
    Evolution<MappingDesc> evo;  //holds rule mask derived from 23/3 input, see Evolution.hpp
    GatherSlice gather;
    
//typedef GridBuffer<uint8_t, DIM2> Buffer; //for storing black and white(live or dead) data for gol
    Buffer* buff1;   //Buffer(see types.h) for swapping between old and new world
    Buffer* buff2;   //like: evolve(buff2 &, const buff1) would work internally
    uint32_t steps;

    bool isMaster;   //why not bool isMaster=false; but in constructor?

    #if DIFF_ADDON == 1
        Buffer* bufdif;      //stores what changed from one timestep to next
        MappingDesc mappingdiff;
    #endif

public:

    Simulation(uint32_t rule, int32_t steps, Space gridSize, Space devices, Space periodic) :
    evo(rule), steps(steps), gridSize(gridSize), isMaster(false), buff1(NULL), buff2(NULL)
    {
       //first call to getInstance initializes SingletonClass Object (???)
       //As this is used to manage the grid of mpi processes, first arg is devices dim
       Environment<DIM2>::get().initDevices(devices, periodic);

       GridController<DIM2> & gc = Environment<DIM2>::get().GridController(); 
       Space localGridSize(gridSize / devices);  //typedef DataSpace<DIM2> Space;
       //Initialize global Variable/Singleton with gridSize/devices and global simulation size:
       //  gridSize and Offset as calculated from GridController-Position
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
        #if DIFF_ADDON == 1
            __delete(bufdif);
        #endif
    }

    void init()
    {
        /* copy singleton simulationBox data to simbox (#define would do the same)  *
         * simBox holds global and local SimulationSize and where the local SimArea *
         * is in the greater scheme using Offsets from global LEFT,TOP,FRONT        *
         * global variable/singleton is set by constructor of this class            */
        PMACC_AUTO(simBox, Environment<DIM2>::get().SubGrid().getSimulationBox());
        
        /* Recall that in types.hpp following is defined:                           *
         *     typedef MappingDescription<DIM2, TVec<16,16> > MappingDesc;          *
         * where TVec<16,16> is arbitrarily(!) chosen SuperCellSize and DIM2 is the *
         * dimension of the grid                                                    *
         * Expression of 2nd argument translates to DataSpace<DIM3>(16,16,0). This  *
         * is the guard-size (here set to be one Supercell wide in all directions). *
         * Meaning we have 16*16*(2*grid.x+2*grid.y+4) more cells in GridLayout     *
         * than in SimulationBox.                                                   */
        GridLayout<DIM2> layout( simBox.getLocalSize(), MappingDesc::SuperCellSize::getDataSpace());
        
        /* getDataSpace will return DataSpace( grid.x +16+16, grid.y +16+16)        */
        evo.init(MappingDesc(layout.getDataSpace(), 1, 1));
        /* Following does the same for the Addon as the above line does for the Core*/ //(!!! explain mapping Desc)
        #if DIFF_ADDON == 1
            mappingdiff = MappingDesc( layout.getDataSpace(), 1, 1 );
            bufdif = new Buffer(layout, true);
        #endif
        
        /* TODO: I think CommunicationTag was misused here. Second argument is bool *
         * whether size is also stored on device! Not sure why size will be needed  *
         * Before: buff2 = new Buffer(layout, BUFF2);                               */
        buff1 = new Buffer(layout, true);
        buff2 = new Buffer(layout, false);

        /*DataSpace<DIM2>*/ Space guardingCells(1, 1);
        for (uint32_t i = 1; i <= 9; ++i)
        {
            /* TODO: Here here the directions are actually like one would imagine:  *
             * bit 1 to 9 set or unset. (Note that bit 0 is forgotten/unused ...    *
             * Where 1 to 9 represent: left,right,bottom,top,lefttop,leftbottom,... *
             * It's not clear which number corresponds to which direction           *
             * In 3D this would 26 directions. In 2D it would be 8. Don't know why  *
             * 9 directions are initialized ...                                     *
             * types.hpp: enum CommunicationTags{ BUFF1 = 0u, BUFF2 = 1u };         */
            buff1->addExchange(GUARD, Mask(i), guardingCells, BUFF1);
            buff2->addExchange(GUARD, Mask(i), guardingCells, BUFF2);
            /* The addon doesn't need this line, because neighbors don't matter     */
            //bufdif->addExchange(GUARD, Mask(i), guardingCells, 3);
        }
        MessageHeader header(gridSize, layout, simBox.getGlobalOffset());
        isMaster = gather.init(header, true);

        /* Calls kernel to initialize random generator and games of life world      */
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
        /* TODO: what exactly is being transmitted how from where to where?         */
        PMACC_AUTO(splitEvent, __getTransactionEvent());

        /* communication is asynchron to the next tasks */
        PMACC_AUTO(send, read->asyncCommunication(splitEvent));
        evo.run<CORE > (read->getDeviceBuffer().getDataBox(), write->getDeviceBuffer().getDataBox());

        /* Join communication with worker tasks, Now all next tasks run sequential  */
        __setTransactionEvent(send);

        evo.run<BORDER > (read->getDeviceBuffer().getDataBox(), write->getDeviceBuffer().getDataBox());

        write->deviceToHost();

        /* We need PMACC_AUTO here, because DataBox<base>, base is not easily known.*
         * In this case we know, that base is except one case in PiConGPU           *
         * PitchedBox<Type,Dim>. But we would have to derive these two template     *
         * parameters for PitchedBox from steps waaay before this one               *
         *   DataBox<PitchedBox<?,DIM2>> picture = ...                              */
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
                        ( mapper.getGridDim(), MappingDesc::SuperCellSize::getDataSpace())
                        ( buff1->getDeviceBuffer().getDataBox(),
                          buff2->getDeviceBuffer().getDataBox(),
                          bufdif->getDeviceBuffer().getDataBox(),
                          mapper);
            //you can comment this out to see exactly what the Borders are. They will appear as black spaces/margins arougn the cores
            AreaMapping<BORDER,MappingDesc> mapper2(mappingdiff);
            __cudaKernel( gol::kernel::diffEvolution )
                        ( mapper2.getGridDim(), MappingDesc::SuperCellSize::getDataSpace())
                        ( buff1->getDeviceBuffer().getDataBox(),
                          buff2->getDeviceBuffer().getDataBox(),
                          bufdif->getDeviceBuffer().getDataBox(),
                          mapper2);
            bufdif->deviceToHost();
            PMACC_AUTO( difpic, gather( bufdif->getHostBuffer().getDataBox() ) );
            PngCreator pngdiff;
            if (isMaster) { //watch out for race conditions when writing to the same file!
                pngdiff(currentStep, difpic, gridSize, "dif/dif");
            }
            std::stringstream filename2;
            GridController<DIM2> & gc = Environment<DIM2>::get().GridController(); 
            PMACC_AUTO(simBox, Environment<DIM2>::get().SubGrid().getSimulationBox());
            filename2
               << "dif/partial/Rank-" << std::setw(3) << std::setfill('0') << gc.getGlobalRank()
               << "_x-Pos-" << std::setw(3) << std::setfill('0') << simBox.getGlobalOffset()[0]
               << "_y-Pos-" << std::setw(3) << std::setfill('0') << simBox.getGlobalOffset()[1] << "\0";
            pngdiff( currentStep, bufdif->getHostBuffer().getDataBox(), bufdif->getGridLayout().getDataSpace(), filename2.str() );
            
            //TODO: Describe Debug Output done here
            PngCreator pngpartial;
            std::stringstream filename;
            filename 
               << "partial/Rank-" << std::setw(3) << std::setfill('0') << gc.getGlobalRank()
               << "_x-Pos-" << std::setw(3) << std::setfill('0') << simBox.getGlobalOffset()[0]
               << "_y-Pos-" << std::setw(3) << std::setfill('0') << simBox.getGlobalOffset()[1] << "\0";
            pngpartial( currentStep, write->getHostBuffer().getDataBox(), write->getGridLayout().getDataSpace(), filename.str() );
        #endif
    }


    /**************************************************************************
     * test whether cuda device with deviceNumber exists and is usable.       *
     * If so, set this process to use only that device from now on, affecting *
     * cudaMalloc,cudaKernelLaunch, ...                                       *
     **************************************************************************/
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
            //connect CUDA device with thread. Thread 1 therefore will only use cudaDevice 1
            CUDA_CHECK(rc = cudaSetDevice(deviceNumber));
            if (cudaSuccess == rc)
            {
                cudaDeviceProp dprop;
                cudaGetDeviceProperties(&dprop, deviceNumber);
                //!\todo: write this only on debug //TODO: where does log go to ?
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

