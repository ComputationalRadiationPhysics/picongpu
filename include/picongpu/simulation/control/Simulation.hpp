/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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

#include <pmacc/verify.hpp>
#include <pmacc/assert.hpp>

#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/mpl/count.hpp>

#include <pmacc/types.hpp>
#include <pmacc/simulationControl/SimulationHelper.hpp>
#include "picongpu/simulation_defines.hpp"
#include "picongpu/versionFormat.hpp"
#include "picongpu/random/seed/ISeed.hpp"

#include <pmacc/eventSystem/EventSystem.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/nvidia/memory/MemoryInfo.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include "picongpu/simulation/control/MovingWindow.hpp"
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>

#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/fields/MaxwellSolver/YeePML/Field.hpp"
#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/initialization/IInitPlugin.hpp"
#include "picongpu/initialization/ParserGridDistribution.hpp"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/manipulators/manipulators.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/flylite/NonLTE.tpp"
#include "picongpu/simulation/control/DomainAdjuster.hpp"
#include "picongpu/simulation/stage/Bremsstrahlung.hpp"
#include "picongpu/simulation/stage/CurrentBackground.hpp"
#include "picongpu/simulation/stage/CurrentDeposition.hpp"
#include "picongpu/simulation/stage/CurrentInterpolationAndAdditionToEMF.hpp"
#include "picongpu/simulation/stage/CurrentReset.hpp"
#include "picongpu/simulation/stage/FieldBackground.hpp"
#include "picongpu/simulation/stage/MomentumBackup.hpp"
#include "picongpu/simulation/stage/ParticleIonization.hpp"
#include "picongpu/simulation/stage/ParticlePush.hpp"
#include "picongpu/simulation/stage/PopulationKinetics.hpp"
#include "picongpu/simulation/stage/SynchrotronRadiation.hpp"
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/random/RNGProvider.hpp>

#if(PMACC_CUDA_ENABLED == 1)
#    include "picongpu/particles/bremsstrahlung/ScaledSpectrum.hpp"
#    include "picongpu/particles/bremsstrahlung/PhotonEmissionAngle.hpp"
#endif

#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.hpp"

#include <pmacc/nvidia/reduce/Reduce.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/meta/conversion/SeqToMap.hpp>
#include <pmacc/meta/conversion/TypeToPointerPair.hpp>

#include <pmacc/meta/ForEach.hpp>
#include "picongpu/particles/ParticlesFunctors.hpp"
#include "picongpu/particles/InitFunctors.hpp"
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/particles/traits/FilterByIdentifier.hpp>
#include <pmacc/particles/IdProvider.hpp>

#include <boost/mpl/int.hpp>
#include <memory>
#include <functional>


namespace picongpu
{
    using namespace pmacc;

    /**
     * Global simulation controller class.
     *
     * Initialises simulation data and defines the simulation steps
     * for each iteration.
     *
     * @tparam DIM the dimension (2-3) for the simulation
     */
    class Simulation : public SimulationHelper<simDim>
    {
    public:
        /**
         * Constructor
         */
        Simulation()
            : myFieldSolver(nullptr)
            , cellDescription(nullptr)
            , initialiserController(nullptr)
            , slidingWindow(false)
            , windowMovePoint(0.0)
            , endSlidingOnStep(-1)
            , showVersionOnce(false)
        {
        }

        virtual void pluginRegisterHelp(po::options_description& desc)
        {
            SimulationHelper<simDim>::pluginRegisterHelp(desc);
            currentInterpolationAndAdditionToEMF.registerHelp(desc);
            fieldBackground.registerHelp(desc);
            desc.add_options()(
                "versionOnce",
                po::value<bool>(&showVersionOnce)->zero_tokens(),
                "print version information once and start")

                ("devices,d",
                 po::value<std::vector<uint32_t>>(&devices)->multitoken(),
                 "number of devices in each dimension")

                    ("grid,g",
                     po::value<std::vector<uint32_t>>(&gridSize)->multitoken(),
                     "size of the simulation grid")

                        ("gridDist",
                         po::value<std::vector<std::string>>(&gridDistribution)->multitoken(),
                         "Regex to describe the static distribution of the cells for each device,"
                         "default: equal distribution over all devices\n"
                         "  example:\n"
                         "    -d 2 4 1\n"
                         "    -g 128 192 12\n"
                         "    --gridDist \"64{2}\" \"64,32{2},64\"\n")

                            ("periodic",
                             po::value<std::vector<uint32_t>>(&periodic)->multitoken(),
                             "specifying whether the grid is periodic (1) or not (0) in each dimension, default: no "
                             "periodic dimensions")

                                ("moving,m",
                                 po::value<bool>(&slidingWindow)->zero_tokens(),
                                 "enable sliding/moving window")
                /* For now we still use the compile-time movePoint variable to set
                 * the default value and provide backward compatibility
                 */
                ("windowMovePoint",
                 po::value<float_64>(&windowMovePoint)->default_value(movePoint),
                 "ratio of the global window size in y which defines when to "
                 "start sliding the window. "
                 "The window starts sliding at the time required to pass the "
                 "distance of windowMovePoint * (global window size in y) "
                 "when moving with the speed of light")(
                    "stopWindow",
                    po::value<int32_t>(&endSlidingOnStep)->default_value(-1),
                    "stops the window at stimulation step, "
                    "-1 means that window is never stopping")(
                    "autoAdjustGrid",
                    po::value<bool>(&autoAdjustGrid)->default_value(true),
                    "auto adjust the grid size if PIConGPU conditions are not fulfilled");
        }

        std::string pluginGetName() const
        {
            return "PIConGPU";
        }

        virtual void pluginLoad()
        {
            // fill periodic with 0
            while(periodic.size() < 3)
                periodic.push_back(0);

            // check on correct number of devices. fill with default value 1 for missing dimensions
            if(devices.size() > 3)
            {
                std::cerr << "Invalid number of devices.\nuse [-d dx=1 dy=1 dz=1]" << std::endl;
            }
            else
                while(devices.size() < 3)
                    devices.push_back(1);

            // check on correct grid size. fill with default grid size value 1 for missing 3. dimension
            if(gridSize.size() < 2 || gridSize.size() > 3)
            {
                std::cerr << "Invalid or missing grid size.\nuse -g width height [depth=1]" << std::endl;
            }
            else if(gridSize.size() == 2)
                gridSize.push_back(1);

            if(slidingWindow && devices[1] == 1)
            {
                std::cerr << "Invalid configuration. Can't use moving window with one device in Y direction"
                          << std::endl;
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
            cellDescription = new MappingDesc(layout.getDataSpace(), DataSpace<simDim>(GuardSize::toRT()));

            if(gc.getGlobalRank() == 0)
            {
                if(MovingWindow::getInstance().isEnabled())
                    log<picLog::PHYSICS>("Sliding Window is ON");
                else
                    log<picLog::PHYSICS>("Sliding Window is OFF");
            }
        }

        virtual void pluginUnload()
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            SimulationHelper<simDim>::pluginUnload();

            __delete(myFieldSolver);

            /** unshare all registered ISimulationData sets
             *
             * @todo can be removed as soon as our Environment learns to shutdown in
             *       a distinct order, e.g. DataConnector before CUDA context
             */
            dc.clean();

            __delete(cellDescription);
        }

        void notify(uint32_t)
        {
        }

        virtual void init()
        {
            namespace nvmem = pmacc::nvidia::memory;

            // This has to be called before initFields()
            currentInterpolationAndAdditionToEMF.init();

            DataConnector& dc = Environment<>::get().DataConnector();
            initFields(dc);

            // create field solver
            this->myFieldSolver = new fields::Solver(*cellDescription);

            // initialize field background stage,
            // this may include allocation of additional fields so has to be done before particles
            fieldBackground.init(*cellDescription);

            // Initialize random number generator and synchrotron functions, if there are synchrotron or bremsstrahlung
            // Photons
            using AllSynchrotronPhotonsSpecies =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, synchrotronPhotons<>>::type;
            using AllBremsstrahlungPhotonsSpecies =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, bremsstrahlungPhotons<>>::type;

            // create factory for the random number generator
            const uint32_t userSeed = random::seed::ISeed<random::SeedGenerator>{}();
            const uint32_t seed = std::hash<std::string>{}(std::to_string(userSeed));

            using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
            auto rngFactory = std::make_unique<RNGFactory>(Environment<simDim>::get().SubGrid().getLocalDomain().size);
            if(Environment<simDim>::get().GridController().getGlobalRank() == 0)
            {
                log<picLog::PHYSICS>("used Random Number Generator: %1% seed: %2%") % rngFactory->getName() % userSeed;
            }

            // init and share random number generator
            pmacc::GridController<simDim>& gridCon = pmacc::Environment<simDim>::get().GridController();
            rngFactory->init(gridCon.getScalarPosition() ^ seed);
            dc.consume(std::move(rngFactory));

            // Initialize synchrotron functions, if there are synchrotron photon species
            if(!bmpl::empty<AllSynchrotronPhotonsSpecies>::value)
            {
                this->synchrotronFunctions.init();
            }
#if(PMACC_CUDA_ENABLED == 1)
            // Initialize bremsstrahlung lookup tables, if there are species containing bremsstrahlung photons
            if(!bmpl::empty<AllBremsstrahlungPhotonsSpecies>::value)
            {
                meta::ForEach<
                    AllBremsstrahlungPhotonsSpecies,
                    particles::bremsstrahlung::FillScaledSpectrumMap<bmpl::_1>>
                    fillScaledSpectrumMap;
                fillScaledSpectrumMap(this->scaledBremsstrahlungSpectrumMap);

                this->bremsstrahlungPhotonAngle.init();
            }
#endif

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
            auto nativeCudaStream = cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(0);
            /* Create an empty allocator. This one is resized after all exchanges
             * for particles are created */
            deviceHeap.reset(

                new DeviceHeap(cupla::manager::Device<cupla::AccDev>::get().current(), nativeCudaStream, 0u));
            cuplaStreamSynchronize(0);
#endif

            /* Allocate helper fields for FLYlite population kinetics for atomic physics
             * (histograms, rate matrix, etc.)
             */
            using AllFlyLiteIons =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, populationKinetics<>>::type;

            meta::ForEach<AllFlyLiteIons, particles::CallPopulationKineticsInit<bmpl::_1>, bmpl::_1>
                initPopulationKinetics;
            initPopulationKinetics(gridSizeLocal);

            // Allocate and initialize particle species with all left-over memory below
            meta::ForEach<VectorAllSpecies, particles::CreateSpecies<bmpl::_1>> createSpeciesMemory;
            createSpeciesMemory(deviceHeap, cellDescription);

            size_t freeGpuMem(0);
            Environment<>::get().MemoryInfo().getMemoryInfo(&freeGpuMem);
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

            if(Environment<>::get().MemoryInfo().isSharedMemoryPool())
            {
                heapSize /= 2;
                log<picLog::MEMORY>(
                    "Shared RAM between GPU and host detected - using only half of the 'device' memory.");
            }
            else
                log<picLog::MEMORY>("RAM is NOT shared between GPU and host.");

            // initializing the heap for particles
            deviceHeap->destructiveResize(
                cupla::manager::Device<cupla::AccDev>::get().current(),
                nativeCudaStream,
                heapSize);
            cuplaStreamSynchronize(0);

            auto mallocMCBuffer = std::make_unique<MallocMCBuffer<DeviceHeap>>(deviceHeap);
            dc.consume(std::move(mallocMCBuffer));

#endif

            meta::ForEach<VectorAllSpecies, particles::LogMemoryStatisticsForSpecies<bmpl::_1>>
                logMemoryStatisticsForSpecies;
            logMemoryStatisticsForSpecies(deviceHeap);

            Environment<>::get().MemoryInfo().getMemoryInfo(&freeGpuMem);
            log<picLog::MEMORY>("free mem after all mem is allocated %1% MiB") % (freeGpuMem / 1024 / 1024);

            IdProvider<simDim>::init();

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
            /* add CUDA streams to the StreamController for concurrent execution */
            Environment<>::get().StreamController().addStreams(6);
#endif
        }

        virtual uint32_t fillSimulation()
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

            DataConnector& dc = Environment<>::get().DataConnector();
            auto fieldE = dc.get<FieldE>(FieldE::getName(), true);
            auto fieldB = dc.get<FieldB>(FieldB::getName(), true);

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
                            throw std::runtime_error("Restart failed. You must provide the "
                                                     "'--checkpoint.restart.step' argument. See picongpu --help.");
                        }
                        else
                            this->restartStep = checkpoints.back();
                    }

                    initialiserController->restart((uint32_t) this->restartStep, this->restartDirectory);
                    step = this->restartStep;
                }
                else
                {
                    initialiserController->init();
                    meta::ForEach<particles::InitPipeline, particles::CallFunctor<bmpl::_1>> initSpecies;
                    initSpecies(step);
                }
            }

            size_t freeGpuMem(0u);
            Environment<>::get().MemoryInfo().getMemoryInfo(&freeGpuMem);
            log<picLog::MEMORY>("free mem after all particles are initialized %1% MiB") % (freeGpuMem / 1024 / 1024);

            // generate valid GUARDS (overwrite)
            EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(eRfieldE);
            EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(eRfieldB);

            dc.releaseData(FieldE::getName());
            dc.releaseData(FieldB::getName());

            return step;
        }

        /**
         * Run one simulation step.
         *
         * @param currentStep iteration number of the current step
         */
        virtual void runOneStep(uint32_t currentStep)
        {
            using namespace simulation::stage;
            MomentumBackup{}(currentStep);
            CurrentReset{}(currentStep);
            ParticleIonization{*cellDescription}(currentStep);
            PopulationKinetics{}(currentStep);
            SynchrotronRadiation{*cellDescription, synchrotronFunctions}(currentStep);
#if(PMACC_CUDA_ENABLED == 1)
            Bremsstrahlung{*cellDescription, scaledBremsstrahlungSpectrumMap, bremsstrahlungPhotonAngle}(currentStep);
#endif
            EventTask commEvent;
            ParticlePush{}(currentStep, commEvent);
            fieldBackground.subtract(currentStep);
            myFieldSolver->update_beforeCurrent(currentStep);
            __setTransactionEvent(commEvent);
            CurrentBackground{*cellDescription}(currentStep);
            CurrentDeposition{}(currentStep);
            currentInterpolationAndAdditionToEMF(currentStep);
            myFieldSolver->update_afterCurrent(currentStep);
        }

        virtual void movingWindowCheck(uint32_t currentStep)
        {
            if(MovingWindow::getInstance().slideInCurrentStep(currentStep))
            {
                slide(currentStep);
            }

            /* do not double-add background field on restarts
             * (contained in checkpoint data)
             */
            bool addBgFields = true;
            if(this->restartRequested)
            {
                if(this->restartStep == int32_t(currentStep))
                    addBgFields = false;
            }

            if(addBgFields)
            {
                /* add background field: the movingWindowCheck is just at the start
                 * of a time step before all the plugins are called (and the step
                 * itself is performed for this time step).
                 * Hence the background field is visible for all plugins
                 * in between the time steps.
                 */
                fieldBackground.add(currentStep);
            }
        }

        virtual void resetAll(uint32_t currentStep)
        {
            resetFields(currentStep);
            meta::ForEach<VectorAllSpecies, particles::CallReset<bmpl::_1>> resetParticles;
            resetParticles(currentStep);
        }

        void slide(uint32_t currentStep)
        {
            GridController<simDim>& gc = Environment<simDim>::get().GridController();

            if(gc.slide())
            {
                log<picLog::SIMULATION_STATE>("slide in step %1%") % currentStep;
                resetAll(currentStep);
                initialiserController->slide(currentStep);
                meta::ForEach<particles::InitPipeline, particles::CallFunctor<bmpl::_1>> initSpecies;
                initSpecies(currentStep);
            }
        }

        virtual void setInitController(IInitPlugin* initController)
        {
            PMACC_ASSERT(initController != nullptr);
            this->initialiserController = initController;
        }

        MappingDesc* getMappingDescription()
        {
            return cellDescription;
        }

    protected:
        std::shared_ptr<DeviceHeap> deviceHeap;

        fields::Solver* myFieldSolver;
        simulation::stage::CurrentInterpolationAndAdditionToEMF currentInterpolationAndAdditionToEMF;

        // Field background stage, has to live always as it is used for registering options like a plugin.
        // Because of it, has a special init() method that has to be called during initialization of the simulation
        simulation::stage::FieldBackground fieldBackground;

#if(PMACC_CUDA_ENABLED == 1)
        // creates lookup tables for the bremsstrahlung effect
        // map<atomic number, scaled bremsstrahlung spectrum>
        std::map<float_X, particles::bremsstrahlung::ScaledSpectrum> scaledBremsstrahlungSpectrumMap;
        particles::bremsstrahlung::GetPhotonAngle bremsstrahlungPhotonAngle;
#endif

        // Synchrotron functions (used in synchrotronPhotons module)
        particles::synchrotronPhotons::SynchrotronFunctions synchrotronFunctions;

        // output classes

        IInitPlugin* initialiserController;

        MappingDesc* cellDescription;

        // layout parameter
        std::vector<uint32_t> devices;
        std::vector<uint32_t> gridSize;
        /** Without guards */
        DataSpace<simDim> gridSizeLocal;
        std::vector<uint32_t> periodic;

        std::vector<std::string> gridDistribution;

        bool slidingWindow;
        int32_t endSlidingOnStep;
        float_64 windowMovePoint;
        bool showVersionOnce;
        bool autoAdjustGrid = true;

    private:
        void initFields(DataConnector& dataConnector)
        {
            auto fieldB = std::make_unique<FieldB>(*cellDescription);
            dataConnector.consume(std::move(fieldB));
            auto fieldE = std::make_unique<FieldE>(*cellDescription);
            dataConnector.consume(std::move(fieldE));
            auto fieldJ = std::make_unique<FieldJ>(*cellDescription);
            dataConnector.consume(std::move(fieldJ));
            for(uint32_t slot = 0; slot < fieldTmpNumSlots; ++slot)
            {
                auto fieldTmp = std::make_unique<FieldTmp>(*cellDescription, slot);
                dataConnector.consume(std::move(fieldTmp));
            }
        }

        /** Reset all fields
         *
         * @param currentStep iteration number of the current step
         */
        void resetFields(uint32_t const currentStep)
        {
            auto resetField = [currentStep](std::string const name) {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto const fieldExists = dc.hasId(name);
                if(fieldExists)
                {
                    using FieldHelper = SimulationFieldHelper<MappingDesc>;
                    auto field = std::dynamic_pointer_cast<FieldHelper>(dc.get<ISimulationData>(name, true));
                    if(field)
                        field->reset(currentStep);
                    dc.releaseData(name);
                }
            };

            /* @todo for now the list of fields is hardcoded here, a more generic
             * solution would require changes to design of DataConnector.
             * FieldJ and FieldTmp are effectively cleared each time iteration and
             * so do not need a reset.
             */
            std::array<std::string, 4> const fieldNames{
                {FieldE::getName(),
                 FieldB::getName(),
                 fields::maxwellSolver::yeePML::FieldE::getName(),
                 fields::maxwellSolver::yeePML::FieldB::getName()}};
            std::for_each(fieldNames.cbegin(), fieldNames.cend(), resetField);
        }
    };
} /* namespace picongpu */

#include "picongpu/fields/Fields.tpp"
#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.tpp"

#if(PMACC_CUDA_ENABLED == 1)
#    include "picongpu/particles/bremsstrahlung/Bremsstrahlung.tpp"
#    include "picongpu/particles/bremsstrahlung/ScaledSpectrum.tpp"
#endif
