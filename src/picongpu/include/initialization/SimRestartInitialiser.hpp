/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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



#ifndef SIMRESTARTINITIALISER_HPP
#define	SIMRESTARTINITIALISER_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "particles/frame_types.hpp"

#include "dataManagement/AbstractInitialiser.hpp"
#include "dataManagement/DataConnector.hpp"

#include "dimensions/DataSpace.hpp"

#include "dimensions/GridLayout.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"

#include <splash/splash.h>

#include "simulationControl/MovingWindow.hpp"

#include <string>
#include <sstream>

namespace picongpu
{

using namespace PMacc;
using namespace splash;

/**
 * Simulation restart initialiser.
 * 
 * Initialises a new simulation from stored data in HDF5 files.
 * 
 * @tparam EBuffer type for Electrons (see MySimulation)
 * @tparam IBuffer type for Ions (see MySimulation)
 * @tparam DIM dimension of the simulation (2-3)
 */
template <class EBuffer, class IBuffer, unsigned DIM>
class SimRestartInitialiser : public AbstractInitialiser
{
public:

    /*! Restart a simulation from a hdf5 dump
     * This class can't restart simulation with active moving (sliding) window
     */
    SimRestartInitialiser(std::string filename, DataSpace<DIM> localGridSize) :
    simulationStep(0),
    localGridSize(localGridSize)
    {
    }

    virtual ~SimRestartInitialiser()
    {
    }

    uint32_t setup()
    {
        // call super class
        AbstractInitialiser::setup();
        
        return 0;
    }

    void teardown()
    {
        // call super class
        AbstractInitialiser::teardown();
    }

    void init(ISimulationData& data, uint32_t)
    {
        SimulationDataId id = data.getUniqueId();
#if 0
#if (ENABLE_ELECTRONS == 1)  
        if (id == EBuffer::FrameType::getName())
        {
            VirtualWindow window = MovingWindow::getInstance().getVirtualWindow(simulationStep);
            DataSpace<simDim> globalDomainOffset(gridPosition);
            DataSpace<simDim> logicalToPhysicalOffset(gridPosition - window.globalSimulationOffset);

            /*domains are allways positiv*/
            if (globalDomainOffset.y() == 0)
                globalDomainOffset.y() = window.globalSimulationOffset.y();

            DataSpace<simDim> localDomainSize(window.localSize);

            log<picLog::INPUT_OUTPUT > ("Begin loading electrons");
            RestartParticleLoader<EBuffer>::loadParticles(simulationStep,
                                                          *dataCollector,
                                                          std::string("particles/") +
                                                          EBuffer::FrameType::getName(),
                                                          static_cast<EBuffer&> (data),
                                                          globalDomainOffset,
                                                          localDomainSize,
                                                          logicalToPhysicalOffset
                                                          );
            log<picLog::INPUT_OUTPUT > ("Finished loading electrons");


            if (MovingWindow::getInstance().isSlidingWindowActive())
            {
                {
                    log<picLog::INPUT_OUTPUT > ("Begin loading electrons bottom");
                    globalDomainOffset = gridPosition;
                    globalDomainOffset.y() += window.localSize.y();

                    localDomainSize = window.localFullSize;
                    localDomainSize.y() -= window.localSize.y();

                    {
                        DataSpace<simDim> particleOffset = gridPosition;
                        particleOffset.y() = -window.localSize.y();
                        RestartParticleLoader<EBuffer>::loadParticles(
                                                                      simulationStep,
                                                                      *dataCollector,
                                                                      std::string("particles/") +
                                                                      EBuffer::FrameType::getName() +
                                                                      std::string("/_ghosts"),
                                                                      static_cast<EBuffer&> (data),
                                                                      globalDomainOffset,
                                                                      localDomainSize,
                                                                      particleOffset
                                                                      );
                    }
                    log<picLog::INPUT_OUTPUT > ("Finished loading electrons bottom");
                }
            }
            return;
        }
#endif
#if (ENABLE_IONS == 1)
        if (id == IBuffer::FrameType::getName())
        {
            VirtualWindow window = MovingWindow::getInstance().getVirtualWindow(simulationStep);
            DataSpace<simDim> globalDomainOffset(gridPosition);
            DataSpace<simDim> logicalToPhysicalOffset(gridPosition - window.globalSimulationOffset);

            /*domains are allways positiv*/
            if (globalDomainOffset.y() == 0)
                globalDomainOffset.y() = window.globalSimulationOffset.y();

            DataSpace<simDim> localDomainSize(window.localSize);

            log<picLog::INPUT_OUTPUT > ("Begin loading ions");
            RestartParticleLoader<IBuffer>::loadParticles(
                                                          simulationStep,
                                                          *dataCollector,
                                                          std::string("particles/") +
                                                          IBuffer::FrameType::getName(),
                                                          static_cast<IBuffer&> (data),
                                                          globalDomainOffset,
                                                          localDomainSize,
                                                          logicalToPhysicalOffset
                                                          );
            log<picLog::INPUT_OUTPUT > ("Finished loading ions");
            if (MovingWindow::getInstance().isSlidingWindowActive())
            {
                {
                    log<picLog::INPUT_OUTPUT > ("Begin loading ions bottom");
                    globalDomainOffset = gridPosition;
                    globalDomainOffset.y() += window.localSize.y();

                    localDomainSize = window.localFullSize;
                    localDomainSize.y() -= window.localSize.y();

                    {
                        DataSpace<simDim> particleOffset = gridPosition;
                        particleOffset.y() = -window.localSize.y();
                        RestartParticleLoader<IBuffer>::loadParticles(
                                                                      simulationStep,
                                                                      *dataCollector,
                                                                      std::string("particles/") +
                                                                      IBuffer::FrameType::getName() +
                                                                      std::string("/_ghosts"),
                                                                      static_cast<IBuffer&> (data),
                                                                      globalDomainOffset,
                                                                      localDomainSize,
                                                                      particleOffset
                                                                      );
                    }
                    log<picLog::INPUT_OUTPUT > ("Finished loading ions bottom");
                }
            }
            return;
        }

#endif
#endif
        /*if (id == FieldE::getName())
        {
            initField(static_cast<FieldE&> (data).getGridBuffer(), FieldE::getName());
            return;
        }

        if (id == FieldB::getName())
        {
            initField(static_cast<FieldB&> (data).getGridBuffer(), FieldB::getName());
            //copy field B to Bavg (this is not exact but a good approximation)
            //cloneField(static_cast<FieldB&> (data).getGridBufferBavg(), static_cast<FieldB&> (data).getGridBuffer(), "Bavg");
            //this copy is only needed if we not write Bavg in HDF5 file
            //send all B fields thus simulation are of neighbors is on all gpus
            static_cast<FieldB&> (data).asyncCommunication(__getTransactionEvent()).waitForFinished();
            return;
        }*/
    }

    uint32_t getSimulationStep()
    {
        return simulationStep;
    }

private:

    DataSpace<DIM> localGridSize;
    std::string filename;
    uint32_t simulationStep;
};
}

#endif	/* SIMRESTARTINITIALISER_HPP */

