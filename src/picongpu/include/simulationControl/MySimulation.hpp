/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus
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

#include <cassert>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>

#include "types.h"
#include "simulationControl/SimulationHelper.hpp"
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
#include "initialization/IInitPlugin.hpp"
#include "initialization/ParserGridDistribution.hpp"

#include "nvidia/reduce/Reduce.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "nvidia/functors/Add.hpp"
#include "nvidia/functors/Sub.hpp"

#include "compileTime/conversion/SeqToMap.hpp"
#include "compileTime/conversion/TypeToPointerPair.hpp"

#include "algorithms/ForEach.hpp"
#include "particles/ParticlesFunctors.hpp"
#include <boost/mpl/int.hpp>

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
     * Constructor
     */
    MySimulation() :
    laser(NULL),
    fieldB(NULL),
    fieldE(NULL),
    fieldJ(NULL),
    fieldTmp(NULL),
    cellDescription(NULL),
    initialiserController(NULL),
    slidingWindow(false)
    {
        ForEach<VectorAllSpecies, particles::AssignNull<bmpl::_1>, MakeIdentifier<bmpl::_1>  > setPtrToNull;
        setPtrToNull(forward(particleStorage));
    }

    virtual void pluginRegisterHelp(po::options_description& desc)
    {
        SimulationHelper<simDim>::pluginRegisterHelp(desc);
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

    std::string pluginGetName() const
    {
        return "PIConGPU";
    }

    virtual void pluginLoad()
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

        Environment<simDim>::get().initDevices(gpus, isPeriodic);

        DataSpace<simDim> myGPUpos( Environment<simDim>::get().GridController().getPosition() );

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

        Environment<simDim>::get().initGrids(global_grid_size, gridSizeLocal, gridOffset);

        MovingWindow::getInstance().setSlidingWindow(slidingWindow);

        log<picLog::DOMAINS > ("rank %1%; localsize %2%; localoffset %3%;") %
            myGPUpos.toString() % gridSizeLocal.toString() % gridOffset.toString();

        SimulationHelper<simDim>::pluginLoad();

        GridLayout<SIMDIM> layout(gridSizeLocal, MappingDesc::SuperCellSize::toRT());
        cellDescription = new MappingDesc(layout.getDataSpace(), GUARD_SIZE, GUARD_SIZE);

        checkGridConfiguration(global_grid_size, cellDescription->getGridLayout());

        if (Environment<simDim>::get().GridController().getGlobalRank() == 0)
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

    virtual void pluginUnload()
    {

        SimulationHelper<simDim>::pluginUnload();
        __delete(fieldB);

        __delete(fieldE);

        __delete(fieldJ);

        __delete(fieldTmp);

        __delete(myFieldSolver);

        ForEach<VectorAllSpecies, particles::CallDelete<bmpl::_1> , MakeIdentifier<bmpl::_1> > deleteParticleMemory;
        deleteParticleMemory(forward(particleStorage));

        __delete(laser);
        __delete(pushBGField);
        __delete(currentBGField);
        __delete(cellDescription);
    }

    void notify(uint32_t)
    {

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
        currentBGField = new cellwiseOperation::CellwiseOperation < CORE + BORDER + GUARD > (*cellDescription);

        //std::cout<<"Grid x="<<layout.getDataSpace().x()<<" y="<<layout.getDataSpace().y()<<std::endl;

        laser = new LaserPhysics(cellDescription->getGridLayout());

        ForEach<VectorAllSpecies, particles::CreateSpecies<bmpl::_1>, MakeIdentifier<bmpl::_1> > createSpeciesMemory;
        createSpeciesMemory(forward(particleStorage), cellDescription);

        size_t freeGpuMem(0);
        Environment<>::get().EnvMemoryInfo().getMemoryInfo(&freeGpuMem);
        freeGpuMem -= totalFreeGpuMemory;

        ForEach<VectorAllSpecies, particles::CallCreateParticleBuffer<bmpl::_1>, MakeIdentifier<bmpl::_1> > createParticleBuffer;
        createParticleBuffer(forward(particleStorage), freeGpuMem);

        Environment<>::get().EnvMemoryInfo().getMemoryInfo(&freeGpuMem);
        log<picLog::MEMORY > ("free mem after all mem is allocated %1% MiB") % (freeGpuMem / 1024 / 1024);

        fieldB->init(*fieldE, *laser);
        fieldE->init(*fieldB, *laser);
        fieldJ->init(*fieldE);
        fieldTmp->init();

        // create field solver
        this->myFieldSolver = new fieldSolver::FieldSolver(*cellDescription);


        ForEach<VectorAllSpecies, particles::CallInit<bmpl::_1>, MakeIdentifier<bmpl::_1> > particleInit;
        particleInit(forward(particleStorage), fieldE, fieldB, fieldJ, fieldTmp);


        /* add CUDA streams to the StreamController for concurrent execution */
        Environment<>::get().StreamController().addStreams(6);

        uint32_t step = 0;

        if (initialiserController)
        {
            initialiserController->printInformation();
            if (this->restartRequested)
            {
                /* we do not require --restart-step if a master checkpoint file is found */
                if (this->restartStep < 0)
                {
                    this->restartStep = readCheckpointMasterFile();

                    if (this->restartStep < 0)
                    {
                        throw std::runtime_error(
                                "Restart failed. You must provide the '--restart-step' argument. See picongpu --help.");
                    }
                }

                initialiserController->restart((uint32_t)this->restartStep, this->restartDirectory);
                step = this->restartStep + 1;
            }
            else
            {
                initialiserController->init();
            }
        }


        Environment<>::get().EnvMemoryInfo().getMemoryInfo(&freeGpuMem);
        log<picLog::MEMORY > ("free mem after all particles are initialized %1% MiB") % (freeGpuMem / 1024 / 1024);

		/** add background field for particle pusher at the beginning of each simulation, but not at restarts. At restarts the external fields already exists. */
		if (step == 0) {
			namespace nvfct = PMacc::nvidia::functors;
			(*pushBGField)(fieldE, nvfct::Add(), fieldBackground::fieldBackgroundE(fieldE->getUnit()),
						   0, fieldBackground::fieldBackgroundE::InfluenceParticlePusher);
			(*pushBGField)(fieldB, nvfct::Add(), fieldBackground::fieldBackgroundB(fieldB->getUnit()),
						   0, fieldBackground::fieldBackgroundB::InfluenceParticlePusher);
		}
		
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
		
        EventTask initEvent = __getTransactionEvent();
        EventTask updateEvent;
        EventTask commEvent;

        ForEach<VectorAllSpecies, particles::CallUpdate<bmpl::_1>, MakeIdentifier<bmpl::_1> > particleUpdate;
        particleUpdate(forward(particleStorage), currentStep, initEvent, forward(updateEvent), forward(commEvent));

        /** remove background field for particle pusher */
        (*pushBGField)(fieldE, nvfct::Sub(), fieldBackground::fieldBackgroundE(fieldE->getUnit()),
                       currentStep, fieldBackground::fieldBackgroundE::InfluenceParticlePusher);
        (*pushBGField)(fieldB, nvfct::Sub(), fieldBackground::fieldBackgroundB(fieldB->getUnit()),
                       currentStep, fieldBackground::fieldBackgroundB::InfluenceParticlePusher);

        this->myFieldSolver->update_beforeCurrent(currentStep);

        fieldJ->clear();

        __setTransactionEvent(updateEvent + commEvent);
        (*currentBGField)(fieldJ, nvfct::Add(), fieldBackground::fieldBackgroundJ(fieldJ->getUnit()),
                          currentStep, fieldBackground::fieldBackgroundJ::activated);
#if (ENABLE_CURRENT == 1)
        ForEach<VectorAllSpecies, ComputeCurrent<bmpl::_1,bmpl::int_<CORE + BORDER> >, MakeIdentifier<bmpl::_1> > computeCurrent;
        computeCurrent(forward(fieldJ),forward(particleStorage), currentStep);
#endif

#if  (ENABLE_CURRENT == 1)
        if(bmpl::size<VectorAllSpecies>::type::value>0)
        {
            EventTask eRecvCurrent = fieldJ->asyncCommunication(__getTransactionEvent());
            fieldJ->addCurrentToE<CORE > ();

            __setTransactionEvent(eRecvCurrent);
            fieldJ->addCurrentToE<BORDER > ();
        }
#endif

        this->myFieldSolver->update_afterCurrent(currentStep);
		
		/** add background field for particle pusher of next step.
		 *  Hence the background field is visible for all plugins
		 *  in between the time steps.
		 */
        (*pushBGField)(fieldE, nvfct::Add(), fieldBackground::fieldBackgroundE(fieldE->getUnit()),
                       currentStep + 1, fieldBackground::fieldBackgroundE::InfluenceParticlePusher);
        (*pushBGField)(fieldB, nvfct::Add(), fieldBackground::fieldBackgroundB(fieldB->getUnit()),
                       currentStep + 1, fieldBackground::fieldBackgroundB::InfluenceParticlePusher);
    }

    virtual void movingWindowCheck(uint32_t currentStep)
    {
        if (MovingWindow::getInstance().slideInCurrentStep(currentStep))
        {
            slide(currentStep);
        }
    }

    void resetAll(uint32_t currentStep)
    {

        fieldB->reset(currentStep);
        fieldE->reset(currentStep);
        ForEach<VectorAllSpecies, particles::CallReset<bmpl::_1>, MakeIdentifier<bmpl::_1> > callReset;
        callReset(forward(particleStorage), currentStep);
    }

    void slide(uint32_t currentStep)
    {
        GridController<simDim>& gc = Environment<simDim>::get().GridController();

        if (gc.slide())
        {
            log<picLog::SIMULATION_STATE > ("slide in step %1%") % currentStep;
            resetAll(currentStep);
            initialiserController->slide(currentStep);
        }
    }

    virtual void setInitController(IInitPlugin *initController)
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

        assert(globalGridSize[i] % MappingDesc::SuperCellSize::toRT()[i] == 0);
        // local size must a devisor of supercell size
        assert(gridSizeLocal[i] % MappingDesc::SuperCellSize::toRT()[i] == 0);
        // local size must be at least 3 supercells (1x core + 2x border)
        // note: size of border = guard_size (in supercells)
        // \todo we have to add the guard_x/y/z for modified supercells here
        assert( (uint32_t) gridSizeLocal[i] / MappingDesc::SuperCellSize::toRT()[i] >= 3 * GUARD_SIZE);
        }
    }

    /**
     * Return the last line of the checkpoint master file if any
     *
     * @return last checkpoint timestep or -1
     */
    int32_t readCheckpointMasterFile(void)
    {
        int32_t lastCheckpointStep = -1;

        const std::string checkpointMasterFile =
            this->restartDirectory + std::string("/") + this->CHECKPOINT_MASTER_FILE;

        if (boost::filesystem::exists(checkpointMasterFile))
        {
            std::ifstream file;
            file.open(checkpointMasterFile.c_str());

            /* read each line, last line will become the returned checkpoint step */
            std::string line;
            while (file)
            {
                std::getline(file, line);

                if (line.size() > 0)
                {
                    try {
                        lastCheckpointStep = boost::lexical_cast<int32_t>(line);
                    } catch( boost::bad_lexical_cast const& )
                    {
                        std::cerr << "Warning: checkpoint master file contains invalid data ("
                                << line << ")" << std::endl;
                        lastCheckpointStep = -1;
                    }
                }
            }

            file.close();
        }

        return lastCheckpointStep;
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
    cellwiseOperation::CellwiseOperation< CORE + BORDER + GUARD >* currentBGField;

    typedef typename SeqToMap<VectorAllSpecies, TypeToPointerPair<bmpl::_1> >::type ParticleStorageMap;
    typedef PMacc::math::MapTuple<ParticleStorageMap> ParticleStorage;

    ParticleStorage particleStorage;

    LaserPhysics *laser;

    // output classes

    IInitPlugin* initialiserController;

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
} /* namespace picongpu */
