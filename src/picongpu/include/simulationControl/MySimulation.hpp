/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Richard Pausch
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



#ifndef MYSIMULATION_HPP
#define	MYSIMULATION_HPP

#include <cassert>
#include <string>
#include <vector>

#include "types.h"
#include "simulationControl/SimulationHelper.hpp"
#include "simulation_classTypes.hpp"
#include "simulation_types.hpp"
#include "simulation_defines.hpp"

#include "eventSystem/EventSystem.hpp"
#include "dimensions/GridLayout.hpp"
#include "fields/LaserPhysics.hpp"
#include "nvidia/memory/MemoryInfo.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/simulation/GridController.hpp"

#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"
#include "fields/MaxwellSolver/Solvers.hpp"
#include "fields/background/cellwiseOperation.hpp"
#include "initialization/IInitModule.hpp"
#include "initialization/ParserGridDistribution.hpp"

#include "particles/Species.hpp"
#include "moduleSystem/Module.hpp"

#include "nvidia/reduce/Reduce.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "nvidia/functors/Add.hpp"
#include "nvidia/functors/Sub.hpp"

namespace picongpu
{
using namespace PMacc;

/**
 * Global simulation controller class.
 * 
 * Initialises simulation data and defines the simulation steps
 * for each iteration.
 * 
 * @tparam DIM the dimension (2-3) for the simulation
 */
class MySimulation : public SimulationHelper<simDim>
{
public:

    /**
     * Constructor.
     *
     * @param globalGridSize DataSpace describing the initial global grid size for the whole simulation
     * @param gpus DataSpace describing the grid of available GPU devices
     */
    MySimulation() : laser(NULL), fieldB(NULL), fieldE(NULL), fieldJ(NULL), fieldTmp(NULL), cellDescription(NULL), initialiserController(NULL), slidingWindow(false)
    {
#if (ENABLE_IONS == 1)
        ions = NULL;
#endif
#if (ENABLE_ELECTRONS == 1)
        electrons = NULL;

#endif
    }

    virtual void moduleRegisterHelp(po::options_description& desc)
    {
        SimulationHelper<simDim>::moduleRegisterHelp(desc);
        desc.add_options()
            ("devices,d", po::value<std::vector<uint32_t> > (&devices)->multitoken(), "number of devices in each dimension")

            ("grid,g", po::value<std::vector<uint32_t> > (&gridSize)->multitoken(),
             "size of the simulation grid")

            ("gridDist", po::value<std::vector<std::string> > (&gridDistribution)->multitoken(),
             "Regex to describe the static distribution of the cells for each GPU,"
             "default: equal distribution over all GPUs\n"
             "  example:\n"
             "    -d 2 4 1\n"
             "    -g 128 192 12\n"
             "    --gridDist \"64{2}\" \"64,32{2},64\"\n")

            ("periodic", po::value<std::vector<uint32_t> > (&periodic)->multitoken(),
             "specifying whether the grid is periodic (1) or not (0) in each dimension, default: no periodic dimensions")

            ("moving,m", po::value<bool>(&slidingWindow)->zero_tokens(), "enable sliding/moving window");
    }

    std::string moduleGetName() const
    {
        return "PIConGPU";
    }

    virtual void moduleLoad()
    {


        //fill periodic with 0
        while (periodic.size() < 3)
            periodic.push_back(0);

        // check on correct number of devices. fill with default value 1 for missing dimensions
        if (devices.size() > 3)
        {
            std::cerr << "Invalid number of devices.\nuse [-d dx=1 dy=1 dz=1]" << std::endl;
        }
        else
            while (devices.size() < 3)
                devices.push_back(1);

        // check on correct grid size. fill with default grid size value 1 for missing 3. dimension
        if (gridSize.size() < 2 || gridSize.size() > 3)
        {
            std::cerr << "Invalid or missing grid size.\nuse -g width height [depth=1]" << std::endl;
        }
        else
            if (gridSize.size() == 2)
            gridSize.push_back(1);

        if (slidingWindow && devices[1] == 1)
        {
            std::cerr << "Invalid configuration. Can't use moving window with one device in Y direction" << std::endl;
        }

        DataSpace<simDim> global_grid_size;
        DataSpace<simDim> gpus;
        DataSpace<simDim> isPeriodic;

        for (uint32_t i = 0; i < simDim; ++i)
        {
            global_grid_size[i] = gridSize[i];
            gpus[i] = devices[i];
            isPeriodic[i] = periodic[i];
        }

        GridController<simDim>::getInstance().init(gpus, isPeriodic);
        DataSpace<simDim> myGPUpos(GridController<simDim>::getInstance().getPosition());

        // calculate the number of local grid cells and
        // the local cell offset to the global box        
        for (uint32_t dim = 0; dim < gridDistribution.size(); ++dim)
        {
            // parse string
            ParserGridDistribution parserGD(gridDistribution.at(dim));

            // calculate local grid points & offset
            gridSizeLocal[dim] = parserGD.getLocalSize(myGPUpos[dim]);
            gridOffset[dim] = parserGD.getOffset(myGPUpos[dim], global_grid_size[dim]);
        }
        // by default: use an equal distributed box for all omitted params
        for (uint32_t dim = gridDistribution.size(); dim < simDim; ++dim)
        {
            gridSizeLocal[dim] = global_grid_size[dim] / gpus[dim];
            gridOffset[dim] = gridSizeLocal[dim] * myGPUpos[dim];
        }

        MovingWindow::getInstance().setGlobalSimSize(global_grid_size);
        MovingWindow::getInstance().setSlidingWindow(slidingWindow);
        MovingWindow::getInstance().setGpuCount(gpus);

        log<picLog::DOMAINS > ("rank %1%; localsize %2%; localoffset %3%;") %
            myGPUpos.toString() % gridSizeLocal.toString() % gridOffset.toString();

        /*init SubGrid for global use*/
        SubGrid<simDim>::getInstance().init(gridSizeLocal, global_grid_size, gridOffset);

        SimulationHelper<simDim>::moduleLoad();

        GridLayout<SIMDIM> layout(gridSizeLocal, MappingDesc::SuperCellSize::getDataSpace());
        cellDescription = new MappingDesc(layout.getDataSpace(), GUARD_SIZE, GUARD_SIZE);

        checkGridConfiguration(global_grid_size, cellDescription->getGridLayout());

        if (GridController<simDim>::getInstance().getGlobalRank() == 0)
        {
            if (slidingWindow)
                log<picLog::PHYSICS > ("Sliding Window is ON");
            else
                log<picLog::PHYSICS > ("Sliding Window is OFF");
        }

        for (uint32_t i = 0; i < simDim; ++i)
        {

            /*
             * absorber must be smaller than local gridsize if direction is periodic.
             * absorber can't go over more than one device.
             */
            if (isPeriodic[i] == 0)
            {
                /*negativ direction*/
                assert((int) ABSORBER_CELLS[i][0] <= (int) cellDescription->getGridLayout().getDataSpaceWithoutGuarding()[i]);
                /*positiv direction*/
                assert((int) ABSORBER_CELLS[i][1] <= (int) cellDescription->getGridLayout().getDataSpaceWithoutGuarding()[i]);
            }
        }

    }

    virtual void moduleUnload()
    {

        SimulationHelper<simDim>::moduleUnload();
        __delete(fieldB);

        __delete(fieldE);

        __delete(fieldJ);

        __delete(fieldTmp);

        __delete(myFieldSolver);

#if (ENABLE_IONS == 1)
        __delete(ions);
#endif
#if (ENABLE_ELECTRONS == 1)
        __delete(electrons);

#endif
        __delete(laser);
    }

    virtual uint32_t init()
    {
        namespace nvmem = PMacc::nvidia::memory;
        // create simulation data such as fields and particles
        fieldB = new FieldB(*cellDescription);
        fieldE = new FieldE(*cellDescription);
        fieldJ = new FieldJ(*cellDescription);
        fieldTmp = new FieldTmp(*cellDescription);
        pushBGField = new cellwiseOperation::CellwiseOperation < CORE + BORDER + GUARD > (*cellDescription);

        //std::cout<<"Grid x="<<layout.getDataSpace().x()<<" y="<<layout.getDataSpace().y()<<std::endl;

        laser = new LaserPhysics(cellDescription->getGridLayout());

#if (ENABLE_IONS == 1)
        ions = new PIC_Ions(cellDescription->getGridLayout(), *cellDescription);
#endif
#if (ENABLE_ELECTRONS == 1)
        electrons = new PIC_Electrons(cellDescription->getGridLayout(), *cellDescription);
#endif

        size_t freeGpuMem(0);
        nvmem::MemoryInfo::getInstance().getMemoryInfo(&freeGpuMem);
        freeGpuMem -= totalFreeGpuMemory;

#if (ENABLE_IONS == 1)
        log<picLog::MEMORY > ("free mem before ions %1% MiB") % (freeGpuMem / 1024 / 1024);
        ions->createParticleBuffer(freeGpuMem * memFractionIons);
#endif
#if (ENABLE_ELECTRONS == 1)
        size_t memElectrons(0);
        nvmem::MemoryInfo::getInstance().getMemoryInfo(&memElectrons);
        memElectrons -= totalFreeGpuMemory;
        log<picLog::MEMORY > ("free mem before electrons %1% MiB") % (memElectrons / 1024 / 1024);
        electrons->createParticleBuffer(freeGpuMem * memFractionElectrons);
#endif

        nvmem::MemoryInfo::getInstance().getMemoryInfo(&freeGpuMem);
        log<picLog::MEMORY > ("free mem after all mem is allocated %1% MiB") % (freeGpuMem / 1024 / 1024);

        fieldB->init(*fieldE, *laser);
        fieldE->init(*fieldB, *laser);
        fieldJ->init(*fieldE);
        fieldTmp->init();

        // create field solver
        this->myFieldSolver = new fieldSolver::FieldSolver(*cellDescription);

#if (ENABLE_ELECTRONS == 1)
        electrons->init(*fieldE, *fieldB, *fieldJ, PAR_ELECTRONS);
#endif

#if (ENABLE_IONS == 1)
        ions->init(*fieldE, *fieldB, *fieldJ, PAR_IONS);
#endif      
        //disabled because of a transaction system bug
        StreamController::getInstance().addStreams(6);

        uint32_t step = 0;

        if (initialiserController)
            step = initialiserController->init();


        nvmem::MemoryInfo::getInstance().getMemoryInfo(&freeGpuMem);
        log<picLog::MEMORY > ("free mem after all particles are initialized %1% MiB") % (freeGpuMem / 1024 / 1024);

        // communicate all fields
        EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldE);
        EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldB);

        return step;
    }

    virtual ~MySimulation()
    {



    }

    /**
     * Run one simulation step.
     *
     * @param currentStep iteration number of the current step
     */
    virtual void runOneStep(uint32_t currentStep)
    {
        namespace nvfct = PMacc::nvidia::functors;

        /** add background field for particle pusher */
        (*pushBGField)(fieldE, nvfct::Add(), fieldBackgroundE(fieldE->getUnit()),
                       currentStep, fieldBackgroundE::InfluenceParticlePusher);
        (*pushBGField)(fieldB, nvfct::Add(), fieldBackgroundB(fieldB->getUnit()),
                       currentStep, fieldBackgroundB::InfluenceParticlePusher);

#if (ENABLE_IONS == 1)
        __startTransaction(__getTransactionEvent());
        //std::cout << "Begin update Ions" << std::endl;
        ions->update(currentStep);
        //std::cout << "End update Ions" << std::endl;
        EventTask eRecvIons = ions->asyncCommunication(__getTransactionEvent());
        EventTask eIons = __endTransaction();
#endif
#if (ENABLE_ELECTRONS == 1)
        __startTransaction(__getTransactionEvent());
        //std::cout << "Begin update Electrons" << std::endl;
        electrons->update(currentStep);
        //std::cout << "End update Electrons" << std::endl;
        EventTask eRecvElectrons = electrons->asyncCommunication(__getTransactionEvent());
        EventTask eElectrons = __endTransaction();
#endif

        /** remove background field for particle pusher */
        (*pushBGField)(fieldE, nvfct::Sub(), fieldBackgroundE(fieldE->getUnit()),
                       currentStep, fieldBackgroundE::InfluenceParticlePusher);
        (*pushBGField)(fieldB, nvfct::Sub(), fieldBackgroundB(fieldB->getUnit()),
                       currentStep, fieldBackgroundB::InfluenceParticlePusher);

        this->myFieldSolver->update_beforeCurrent(currentStep);

        fieldJ->clear();

#if (ENABLE_IONS == 1)
        __setTransactionEvent(eRecvIons + eIons);
#if (ENABLE_CURRENT ==1)
        fieldJ->computeCurrent < CORE + BORDER, PIC_Ions > (*ions, currentStep);
#endif
#endif
#if (ENABLE_ELECTRONS == 1)
        __setTransactionEvent(eRecvElectrons + eElectrons);
#if (ENABLE_CURRENT ==1)
        fieldJ->computeCurrent < CORE + BORDER, PIC_Electrons > (*electrons, currentStep);
#endif
#endif

#if  (ENABLE_IONS==1) ||  (ENABLE_ELECTRONS==1) && (ENABLE_CURRENT ==1)
        EventTask eRecvCurrent = fieldJ->asyncCommunication(__getTransactionEvent());
        fieldJ->addCurrentToE<CORE > ();

        __setTransactionEvent(eRecvCurrent);
        fieldJ->addCurrentToE<BORDER > ();
#endif

        this->myFieldSolver->update_afterCurrent(currentStep);
    }

    virtual void movingWindowCheck(uint32_t currentStep)
    {
        if (MovingWindow::getInstance().getVirtualWindow(currentStep).doSlide)
        {
            slide(currentStep);
            log<picLog::PHYSICS > ("slide in step %1%") % currentStep;
        }
    }

    void resetAll(uint32_t currentStep)
    {

        fieldB->reset(currentStep);
        fieldE->reset(currentStep);
#if (ENABLE_ELECTRONS == 1)
        electrons->reset(currentStep);
#endif

#if (ENABLE_IONS == 1)
        ions->reset(currentStep);
#endif

    }

    void slide(uint32_t currentStep)
    {
        GridController<simDim>& gc = GridController<simDim>::getInstance();

        if (gc.slide())
        {

            resetAll(currentStep);
            initialiserController->slide(currentStep);
        }
    }

    virtual void setInitController(IInitModule *initController)
    {

        assert(initController != NULL);
        this->initialiserController = initController;
    }

    MappingDesc* getMappingDescription()
    {

        return cellDescription;
    }

private:

    template<uint32_t DIM>
    void checkGridConfiguration(DataSpace<DIM> globalGridSize, GridLayout<DIM>)
    {
        
        for(uint32_t i=0;i<simDim;++i)
        {
        // global size must a devisor of supercell size
        // note: this is redundant, while using the local condition below

        assert(globalGridSize[i] % MappingDesc::SuperCellSize::getDataSpace()[i] == 0);
        // local size must a devisor of supercell size
        assert(gridSizeLocal[i] % MappingDesc::SuperCellSize::getDataSpace()[i] == 0);
        // local size must be at least 3 supercells (1x core + 2x border)
        // note: size of border = guard_size (in supercells)
        // \todo we have to add the guard_x/y/z for modified supercells here
        assert( (uint32_t) gridSizeLocal[i] / MappingDesc::SuperCellSize::getDataSpace()[i] >= 3 * GUARD_SIZE);
        }
    }


protected:
    // fields
    FieldB *fieldB;
    FieldE *fieldE;
    FieldJ *fieldJ;
    FieldTmp *fieldTmp;

    // field solver
    fieldSolver::FieldSolver* myFieldSolver;
    cellwiseOperation::CellwiseOperation< CORE + BORDER + GUARD >* pushBGField;

    // particles
#if (ENABLE_IONS == 1)
    PIC_Ions *ions;
#endif
#if (ENABLE_ELECTRONS == 1)
    PIC_Electrons *electrons;
#endif

    LaserPhysics *laser;

    // output classes

    IInitModule* initialiserController;

    MappingDesc* cellDescription;


    // layout parameter
    std::vector<uint32_t> devices;
    std::vector<uint32_t> gridSize;
    /** Without guards */
    DataSpace<simDim> gridSizeLocal;
    DataSpace<simDim> gridOffset;
    std::vector<uint32_t> periodic;

    std::vector<std::string> gridDistribution;

    bool slidingWindow;

};
}

#endif	/* MYSIMULATION_HPP */

