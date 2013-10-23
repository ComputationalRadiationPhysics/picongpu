/**
 * Copyright 2013 René Widera
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
    typedef MappingDescription<DIM2, TVec < 16, 16 > > MappingDesc;
    typedef Evolution<MappingDesc> Evolutiontype;

    Space gridSize;
    Evolutiontype evo;
    GatherSlice gather;

    Buffer* buff1;
    Buffer* buff2;
    uint32_t steps;

    bool isMaster;

public:

    Simulation(uint32_t rule, int32_t steps, Space gridSize, Space devices, Space periodic) :
    evo(rule), steps(steps), gridSize(gridSize), isMaster(false), buff1(NULL), buff2(NULL)
    {
        /*IMPORTANT: this must called at first PMacc function, before any other grid magic can used*/
        GC::getInstance().init(devices, periodic);
        setDevice((int) (GC::getInstance().getHostRank())); //do this after gridcontroller init

        StreamController::getInstance();
        Manager::getInstance();
        TransactionManager::getInstance();

        Space localGridSize(gridSize / devices);
        SubGrid<DIM2>::getInstance().init(localGridSize, gridSize, GC::getInstance().getPosition() * localGridSize);
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
        PMACC_AUTO(simBox, SubGrid<DIM2>::getInstance().getSimulationBox());
        GridLayout<DIM2> layout(simBox.getLocalSize(),
                                MappingDesc::SuperCellSize::getDataSpace());
        evo.init(MappingDesc(layout.getDataSpace(), 1, 1));

        buff1 = new Buffer(layout, BUFF1);
        buff2 = new Buffer(layout, BUFF2);

        Space gardingCells(1, 1);
        for (uint32_t i = 1; i <= 9; ++i)
        {
            buff1->addExchange(GUARD, Mask(i), gardingCells, BUFF1);
            buff2->addExchange(GUARD, Mask(i), gardingCells, BUFF2);
        }
        MessageHeader header(gridSize, layout, simBox.getGlobalOffset());
        isMaster = gather.init(header, true);

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
        PMACC_AUTO(splitEvent, __getTransactionEvent());

        /*communication is asycron to the next tasks*/
        PMACC_AUTO(send, read->asyncCommunication(splitEvent));
        evo.run<CORE > (read->getDeviceBuffer().getDataBox(), write->getDeviceBuffer().getDataBox());

        /*Join communication with worker tasks, Now all next tasks run sequential*/
        __setTransactionEvent(send);

        evo.run<BORDER > (read->getDeviceBuffer().getDataBox(), write->getDeviceBuffer().getDataBox());

        write->deviceToHost();

        PMACC_AUTO(picture, gather(write->getHostBuffer().getDataBox()));
        PngCreator png;
        if (isMaster) png(currentStep, picture, gridSize);


    }

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

