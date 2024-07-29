/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre, Filip Optolowicz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */
#include "picongpu/simulation/control/Simulation.hpp"

#include "picongpu/particles/debyeLength/Check.hpp"
#include "picongpu/simulation/stage/ParticleInit.hpp"

#include <pmacc/functor/Call.hpp>
#include <pmacc/meta/ForEach.hpp>

#include <boost/lexical_cast.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace picongpu
{
    using namespace pmacc;

    void Simulation::init()
    {
        DataConnector& dc = Environment<>::get().DataConnector();

        dc.share(currentInterpolationAndAdditionToEMF);

        // This has to be called before initFields()
        currentInterpolationAndAdditionToEMF->init();

        currentBackground = std::make_shared<simulation::stage::CurrentBackground>(*cellDescription);
        dc.share(currentBackground);

        synchrotronRadiation = std::make_shared<simulation::stage::SynchrotronRadiation>(*cellDescription);

        initFields(dc);

        myFieldSolver = std::make_shared<fields::Solver>(*cellDescription);
        dc.share(myFieldSolver);

        // initialize field background stage,
        // this may include allocation of additional fields so has to be done before particles
        fieldBackground.init(*cellDescription);

        // initialize particle boundaries
        particleBoundaries.init();

        // initialize runtime density file paths
        runtimeDensityFile.init();

        // create factory for the random number generator
        const uint32_t userSeed = random::seed::ISeed<random::SeedGenerator>{}();
        const uint32_t seed = std::hash<std::string>{}(std::to_string(userSeed));

        using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
        auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
        numRNGsPerSuperCell.x() = numFrameSlots;
        /* For each supercell a linear with numFrameSlots rng states will be created.
         * PMacc's RNG factory does not support a class with N states per supercell therefore the x dimension of
         * the buffer will be multiplied by numFrameSlots.
         */
        auto numRngStates = (Environment<simDim>::get().SubGrid().getLocalDomain().size / SuperCellSize::toRT())
            * numRNGsPerSuperCell;
        auto rngFactory = std::make_unique<RNGFactory>(numRngStates);
        if(Environment<simDim>::get().GridController().getGlobalRank() == 0)
        {
            log<picLog::PHYSICS>("used Random Number Generator: %1% seed: %2%") % rngFactory->getName() % userSeed;
        }

        // init and share random number generator
        pmacc::GridController<simDim>& gridCon = pmacc::Environment<simDim>::get().GridController();
        rngFactory->init(gridCon.getScalarPosition() ^ seed);
        dc.consume(std::move(rngFactory));

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
        auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(ITask::TASK_DEVICE)->getAlpakaQueue();
        auto alpakaDevice = manager::Device<ComputeDevice>::get().current();
        /* Create an empty allocator. This one is resized after all exchanges
         * for particles are created */
        deviceHeap = std::make_shared<DeviceHeap>(alpakaDevice, alpakaQueue, 0u);
        alpaka::wait(alpakaQueue);
#endif

        // Allocate and initialize particle species with all left-over memory below
        meta::ForEach<VectorAllSpecies, particles::CreateSpecies<boost::mpl::_1>> createSpeciesMemory;
        createSpeciesMemory(deviceHeap, cellDescription.get());

        size_t freeGpuMem = freeDeviceMemory();
        if(freeGpuMem < reservedGpuMemorySize)
        {
            pmacc::log<picLog::MEMORY>("%1% MiB free memory < %2% MiB required reserved memory")
                % (freeGpuMem / 1024 / 1024) % (reservedGpuMemorySize / 1024 / 1024);
            std::stringstream msg;
            msg << "Cannot reserve " << (reservedGpuMemorySize / 1024 / 1024) << " MiB as there is only "
                << (freeGpuMem / 1024 / 1024) << " MiB free device memory left";
            throw std::runtime_error(msg.str());
        }

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
        size_t heapSize = freeGpuMem - reservedGpuMemorySize;
        GridController<simDim>& gc = Environment<simDim>::get().GridController();
        if(Environment<>::get().MemoryInfo().isSharedMemoryPool(numRanksPerDevice, gc.getCommunicator().getMPIComm()))
        {
            heapSize /= 2u;
            log<picLog::MEMORY>("Shared RAM between GPU and host detected - using only half of the 'device' memory.");
        }
        else
            log<picLog::MEMORY>("Device RAM is NOT shared between GPU and host.");

        // initializing the heap for particles
        deviceHeap->destructiveResize(alpakaDevice, alpakaQueue, heapSize);
        alpaka::wait(alpakaQueue);

        auto mallocMCBuffer = std::make_unique<MallocMCBuffer<DeviceHeap>>(deviceHeap);
        dc.consume(std::move(mallocMCBuffer));

#endif

        meta::ForEach<VectorAllSpecies, particles::LogMemoryStatisticsForSpecies<boost::mpl::_1>>
            logMemoryStatisticsForSpecies;
        logMemoryStatisticsForSpecies(deviceHeap);

        if(picLog::log_level & picLog::MEMORY::lvl)
        {
            freeGpuMem = freeDeviceMemory();
            log<picLog::MEMORY>("free mem after all mem is allocated %1% MiB") % (freeGpuMem / 1024 / 1024);
        }
        IdProvider<simDim>::init();

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
        /* add CUDA streams to the QueueController for concurrent execution */
        Environment<>::get().QueueController().addQueues(6);
#endif
    }

    uint32_t Simulation::fillSimulation()
    {
        /* assume start (restart in initialiserController might change that) */
        uint32_t step = 0;

        /* set slideCounter properties for PIConGPU MovingWindow: assume start
         * (restart in initialiserController might change this again)
         */
        MovingWindow::getInstance().setSlideCounter(0, 0);
        /* Update MPI domain decomposition: will also update SubGrid domain
         * information such as local offsets in y-direction
         */
        GridController<simDim>& gc = Environment<simDim>::get().GridController();
        gc.setStateAfterSlides(0);

        /* fill all objects registed in DataConnector */
        if(initialiserController)
        {
            initialiserController->printInformation();
            if(this->restartRequested)
            {
                /* we do not require '--checkpoint.restart.step' if a master checkpoint file is found */
                if(this->restartStep < 0)
                {
                    std::vector<uint32_t> checkpoints = readCheckpointMasterFile();

                    if(checkpoints.empty())
                    {
                        if(this->tryRestart == false)
                        {
                            throw std::runtime_error("Restart failed. You must provide the "
                                                     "'--checkpoint.restart.step' argument. See picongpu --help.");
                        }
                        else
                        {
                            // no checkpoint found: start simulation from scratch
                            this->restartRequested = false;
                        }
                    }
                    else
                        this->restartStep = checkpoints.back();
                }
            }

            if(this->restartRequested)
            {
                initialiserController->restart((uint32_t) this->restartStep, this->restartDirectory);
                step = this->restartStep;
            }
            else
            {
                initialiserController->init();
                simulation::stage::ParticleInit{}(step);
                // Check Debye resolution
                particles::debyeLength::check();
            }
        }

        if(picLog::log_level & picLog::MEMORY::lvl)
        {
            size_t freeGpuMem = freeDeviceMemory();
            log<picLog::MEMORY>("free mem after all particles are initialized %1% MiB") % (freeGpuMem / 1024 / 1024);
        }

        DataConnector& dc = Environment<>::get().DataConnector();
        auto fieldE = dc.get<FieldE>(FieldE::getName());
        auto fieldB = dc.get<FieldB>(FieldB::getName());

        // generate valid GUARDS (overwrite)
        EventTask eRfieldE = fieldE->asyncCommunication(eventSystem::getTransactionEvent());
        eventSystem::setTransactionEvent(eRfieldE);
        EventTask eRfieldB = fieldB->asyncCommunication(eventSystem::getTransactionEvent());
        eventSystem::setTransactionEvent(eRfieldB);

        return step;
    }

    void Simulation::slide(uint32_t currentStep)
    {
        GridController<simDim>& gc = Environment<simDim>::get().GridController();

        if(gc.slide())
        {
            log<picLog::SIMULATION_STATE>("slide in step %1%") % currentStep;
            resetAll(currentStep);
            initialiserController->slide(currentStep);
            simulation::stage::ParticleInit{}(currentStep);
        }
    }

    void Simulation::resetAll(uint32_t currentStep)
    {
        resetFields(currentStep);
        meta::ForEach<VectorAllSpecies, particles::CallReset<boost::mpl::_1>> resetParticles;
        resetParticles(currentStep);
    }
    void Simulation::pluginLoad()
    {
        // fill periodic with 0
        while(periodic.size() < 3)
            periodic.push_back(0);


        PMACC_VERIFY_MSG(
            devices.size() >= 2 && devices.size() <= 3,
            "Invalid number of devices.\nuse [-d dx=1 dy=1 dz=1]");

        // check on correct number of devices. fill with default value 1 for missing dimensions
        while(devices.size() < 3)
            devices.push_back(1);

        // check for request of > 1 device in z for a 2d simulation, this is probably a user's mistake
        if((simDim == 2) && (devices[2] > 1))
            std::cerr << "Warning: " << devices[2] << " devices requested for z in a 2d simulation, this parameter "
                      << "will be reset to 1. Number of MPI ranks must be equal to the number of devices in x * y\n";


        PMACC_VERIFY_MSG(
            gridSize.size() >= 2 && gridSize.size() <= 3,
            "Invalid or missing grid size.\nuse -g width height [depth=1]");

        // check on correct grid size. fill with default grid size value 1 for missing 3. dimension
        if(gridSize.size() == 2)
            gridSize.push_back(1);

        if(slidingWindow && devices[1] == 1)
        {
            std::cerr << "Invalid configuration. Can't use moving window with one device in Y direction" << std::endl;
        }

        DataSpace<simDim> gridSizeGlobal;
        DataSpace<simDim> gpus;
        DataSpace<simDim> isPeriodic;

        for(uint32_t i = 0; i < simDim; ++i)
        {
            gridSizeGlobal[i] = gridSize[i];
            gpus[i] = devices[i];
            isPeriodic[i] = periodic[i];
        }

        Environment<simDim>::get().initDevices(gpus, isPeriodic);
        pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();

        DataSpace<simDim> myGPUpos(gc.getPosition());

        if(gc.getGlobalRank() == 0)
        {
            if(showVersionOnce)
            {
                void(getSoftwareVersions(std::cout));
            }
        }

        PMACC_VERIFY_MSG(
            gridDistribution.size() <= 3,
            "Too many grid distribution directions given. A maximum of three directions are supported.");

        // calculate the number of local grid cells and
        // the local cell offset to the global box
        for(uint32_t dim = 0; dim < gridDistribution.size() && dim < simDim; ++dim)
        {
            // parse string
            ParserGridDistribution parserGD(gridDistribution.at(dim));

            // verify number of blocks and devices in dimension match
            parserGD.verifyDevices(gpus[dim]);

            // calculate local grid points & offset
            gridSizeLocal[dim] = parserGD.getLocalSize(myGPUpos[dim]);
        }
        // by default: use an equal distributed box for all omitted params
        for(uint32_t dim = gridDistribution.size(); dim < simDim; ++dim)
        {
            gridSizeLocal[dim] = gridSizeGlobal[dim] / gpus[dim];
        }

        // Absorber has to be loaded before the domain adjuster runs.
        // This is because domain adjuster uses absorber size
        fieldAbsorber.load();

        DataSpace<simDim> gridOffset;

        DomainAdjuster domainAdjuster(gpus, myGPUpos, isPeriodic, slidingWindow);

        if(!autoAdjustGrid)
            domainAdjuster.validateOnly();

        domainAdjuster(gridSizeGlobal, gridSizeLocal, gridOffset);

        Environment<simDim>::get().initGrids(gridSizeGlobal, gridSizeLocal, gridOffset);

        if(!slidingWindow)
        {
            windowMovePoint = 0.0;
            endSlidingOnStep = 0;
        }
        MovingWindow::getInstance().setMovePoint(windowMovePoint);
        MovingWindow::getInstance().setEndSlideOnStep(endSlidingOnStep);

        log<picLog::DOMAINS>("rank %1%; localsize %2%; localoffset %3%;") % myGPUpos.toString()
            % gridSizeLocal.toString() % gridOffset.toString();

        SimulationHelper<simDim>::pluginLoad();

        GridLayout<simDim> layout(gridSizeLocal, GuardSize::toRT() * SuperCellSize::toRT());
        cellDescription = std::make_unique<MappingDesc>(layout.sizeND(), DataSpace<simDim>(GuardSize::toRT()));

        if(gc.getGlobalRank() == 0)
        {
            if(MovingWindow::getInstance().isEnabled())
                log<picLog::PHYSICS>("Sliding Window is ON");
            else
                log<picLog::PHYSICS>("Sliding Window is OFF");
        }
        // doc-include-start: metadata pluginLoad
        addMetadataOf(*this);
        // doc-include-end: metadata pluginLoad
    }

} /* namespace picongpu */
