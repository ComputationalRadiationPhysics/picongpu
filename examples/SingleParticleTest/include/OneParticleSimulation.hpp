/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
 


#ifndef ONEPARTICLESIMULATION_HPP
#define	ONEPARTICLESIMULATION_HPP

#include "simulation_defines.hpp"
#include "Environment.hpp"
#include "simulationControl/MySimulation.hpp"

#include "simulationControl/SimulationHelper.hpp"
#include "simulation_classTypes.hpp"



#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"


#include "dimensions/GridLayout.hpp"
#include "simulation_types.hpp"
#include "eventSystem/EventSystem.hpp"
#include "fields/LaserPhysics.hpp"

#include "nvidia/memory/MemoryInfo.hpp"
#include "mappings/kernel/MappingDescription.hpp"

#include <assert.h>

#include "particles/Species.hpp"

#include "plugins/PluginController.hpp"

#include "particles/ParticlesInitOneParticle.hpp"

#include "particles/Species.hpp"


namespace picongpu
{

using namespace PMacc;

class OneParticleSimulation : public MySimulation
{
public:

    OneParticleSimulation() :
    MySimulation()
    {
    }

    virtual uint32_t init()
    {

        MySimulation::init();

        if (Environment<simDim>::get().GridController().getGlobalRank() == 0)
        {
            std::cout << "max weighting " << NUM_EL_PER_PARTICLE << std::endl;
            std::cout << "courant=min(deltaCellSize)/dt/c > 1.77 ? " << std::min(CELL_WIDTH, std::min(CELL_DEPTH, CELL_HEIGHT)) / SPEED_OF_LIGHT / DELTA_T << std::endl;

#if (LASER_TYPE==1)
            const float_X y_R = M_PI * laserProfile::W0 * laserProfile::W0 / laserProfile::WAVE_LENGTH; //rayleigh length (in y-direction)
            std::cout << "focus/y_Rayleigh: " << laserProfile::FOCUS_POS / y_R << std::endl;
#endif

        }


        //diabled because we have a transaction bug 
        //StreamController::getInstance().addStreams(6);

        //add one particle in simulation
        //
        PMACC_AUTO(simBox, Environment<simDim>::get().SubGrid().getSimulationBox());

        const DataSpace<simDim> halfSimSize(simBox.getGlobalSize() / 2);

        GridLayout<DIM3> layout(simBox.getLocalSize(), MappingDesc::SuperCellSize::getDataSpace());
        MappingDesc cellDescription = MappingDesc(layout.getDataSpace(), GUARD_SIZE, GUARD_SIZE);

        DataSpace<DIM3> centerXZPlan(halfSimSize);
        centerXZPlan.y() = OneParticleOffset; //VACUUM_Y / CELL_HEIGHT + 32; //MappingDesc::SuperCellSize::y + 32;

        ParticlesInitOneParticle<PIC_Electrons>::addOneParticle(*(this->electrons),
                                                                cellDescription,
                                                                centerXZPlan);


        //set E field
        //
        float3_X tmpE;
        tmpE.x() = E_X;
        tmpE.y() = E_Y;
        tmpE.z() = E_Z;
        this->fieldE->getGridBuffer().getDeviceBuffer().setValue(tmpE);

        //set B field
        //
        float3_X tmpB;
        tmpB.x() = B_X;
        tmpB.y() = B_Y;
        tmpB.z() = B_Z;
        this->fieldB->getGridBuffer().getDeviceBuffer().setValue(tmpB);


        return 0;

    }

    /**
     * Run one simulation step.
     *
     * @param currentStep iteration number of the current step
     */
    virtual void runOneStep(uint32_t currentStep)
    {
        fieldJ->clear();
        __startTransaction(__getTransactionEvent());
        //std::cout << "Begin update Electrons" << std::endl;
        electrons->update(currentStep);
        //std::cout << "End update Electrons" << std::endl;
        EventTask eRecvElectrons = electrons->asyncCommunication(__getTransactionEvent());
        EventTask eElectrons = __endTransaction();

        __setTransactionEvent(eRecvElectrons + eElectrons);

#if (ENABLE_CURRENT == 1)
        fieldJ->computeCurrent < CORE + BORDER, PIC_Electrons > (*electrons, currentStep);
#endif

    }

    virtual void movingWindowCheck(uint32_t currentStep)
    {
        PMACC_AUTO(simBox, Environment<simDim>::get().SubGrid().getSimulationBox());
        GridLayout<DIM3> gridLayout(simBox.getLocalSize(), MappingDesc::SuperCellSize::getDataSpace());
        if (MovingWindow::getInstance().getVirtualWindow(currentStep).doSlide)
        {
            GridController<simDim>& gc = Environment<simDim>::get().GridController();
            if (gc.slide())
            {
                electrons->reset(currentStep);
                //set E field
                //
                float3_X tmpE;
                tmpE.x() = E_X;
                tmpE.y() = E_Y;
                tmpE.z() = E_Z;
                this->fieldE->getGridBuffer().getDeviceBuffer().setValue(tmpE);

                //set B field
                //
                float3_X tmpB;
                tmpB.x() = B_X;
                tmpB.y() = B_Y;
                tmpB.z() = B_Z;
                this->fieldB->getGridBuffer().getDeviceBuffer().setValue(tmpB);

                std::cout << "slide" << std::endl;
            }
        }
    }

};

} // namespace picongpu

#endif	/* ONEPARTICLESIMULATION_HPP */

