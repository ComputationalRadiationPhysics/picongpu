/* Copyright 2013-2017 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund
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

#include "verify.hpp"
#include "assert.hpp"

#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/mpl/count.hpp>

#include "pmacc_types.hpp"
#include "simulationControl/SimulationHelper.hpp"
#include "simulation_defines.hpp"

#include "particles/bremsstrahlung/ScaledSpectrum.hpp"
#include "particles/bremsstrahlung/PhotonEmissionAngle.hpp"

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
#include "fields/currentInterpolation/CurrentInterpolation.hpp"
#include "fields/background/cellwiseOperation.hpp"
#include "initialization/IInitPlugin.hpp"
#include "initialization/ParserGridDistribution.hpp"
#include "particles/synchrotronPhotons/SynchrotronFunctions.hpp"
#include "particles/Manipulate.hpp"
#include "particles/manipulators/manipulators.hpp"
#include "random/methods/XorMin.hpp"
#include "random/RNGProvider.hpp"

#include "nvidia/reduce/Reduce.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "nvidia/functors/Add.hpp"
#include "nvidia/functors/Sub.hpp"

#include "compileTime/conversion/SeqToMap.hpp"
#include "compileTime/conversion/TypeToPointerPair.hpp"

#include "algorithms/ForEach.hpp"
#include "particles/ParticlesFunctors.hpp"
#include "particles/InitFunctors.hpp"
#include "particles/memory/buffers/MallocMCBuffer.hpp"
#include "particles/traits/FilterByFlag.hpp"
#include "particles/traits/FilterByIdentifier.hpp"
#include "particles/traits/HasIonizersWithRNG.hpp"
#include "particles/IdProvider.hpp"

#include <boost/mpl/int.hpp>
#include <memory>


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
    laser(nullptr),
    myFieldSolver(nullptr),
    myCurrentInterpolation(nullptr),
    pushBGField(nullptr),
    currentBGField(nullptr),
    cellDescription(nullptr),
    initialiserController(nullptr),
    slidingWindow(false)
    {
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

        DataSpace<simDim> myGPUpos(Environment<simDim>::get().GridController().getPosition());

        // calculate the number of local grid cells and
        // the local cell offset to the global box
        for (uint32_t dim = 0; dim < gridDistribution.size() && dim < simDim; ++dim)
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
                PMACC_VERIFY((int) ABSORBER_CELLS[i][0] <= (int) cellDescription->getGridLayout().getDataSpaceWithoutGuarding()[i]);
                /*positiv direction*/
                PMACC_VERIFY((int) ABSORBER_CELLS[i][1] <= (int) cellDescription->getGridLayout().getDataSpaceWithoutGuarding()[i]);
            }
        }
    }

    virtual void pluginUnload()
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        SimulationHelper<simDim>::pluginUnload();

        __delete(myFieldSolver);

        __delete(myCurrentInterpolation);

        /** unshare all registered ISimulationData sets
         *
         * @todo can be removed as soon as our Environment learns to shutdown in
         *       a distinct order, e.g. DataConnector before CUDA context
         */
        dc.clean();

        __delete(laser);
        __delete(pushBGField);
        __delete(currentBGField);
        __delete(cellDescription);
    }

    void notify(uint32_t)
    {

    }

    virtual void init()
    {
        namespace nvmem = PMacc::nvidia::memory;

        DataConnector &dc = Environment<>::get().DataConnector();

        // create simulation data such as fields and particles
        auto fieldB = new FieldB( *cellDescription );
        dc.share( std::shared_ptr< ISimulationData >( fieldB ) );
        auto fieldE = new FieldE( *cellDescription );
        dc.share( std::shared_ptr< ISimulationData >( fieldE ) );
        auto fieldJ = new FieldJ( *cellDescription );
        dc.share( std::shared_ptr< ISimulationData >( fieldJ ) );

        std::vector< FieldTmp * > fieldTmp;
        for( uint32_t slot = 0; slot < fieldTmpNumSlots; ++slot)
        {
            auto newFld = new FieldTmp( *cellDescription, slot );
            fieldTmp.push_back( newFld );
            dc.share( std::shared_ptr< ISimulationData >( newFld ) );
        }
        pushBGField = new cellwiseOperation::CellwiseOperation < CORE + BORDER + GUARD > (*cellDescription);
        currentBGField = new cellwiseOperation::CellwiseOperation < CORE + BORDER + GUARD > (*cellDescription);

        laser = new LaserPhysics(cellDescription->getGridLayout());

        // Initialize random number generator and synchrotron functions, if there are synchrotron or bremsstrahlung Photons
        typedef typename PMacc::particles::traits::FilterByFlag<VectorAllSpecies,
                                                                synchrotronPhotons<> >::type AllSynchrotronPhotonsSpecies;
        typedef typename PMacc::particles::traits::FilterByFlag<VectorAllSpecies,
                                                                bremsstrahlungPhotons<> >::type AllBremsstrahlungPhotonsSpecies;

        // ... or if ionization methods need an RNG
        using HasIonizerRNGs = typename traits::HasIonizersWithRNG< VectorAllSpecies >::type;
        constexpr bool hasIonizerRNGs = HasIonizerRNGs::value;

        if( !bmpl::empty<AllSynchrotronPhotonsSpecies>::value ||
            !bmpl::empty<AllBremsstrahlungPhotonsSpecies>::value ||
            hasIonizerRNGs
        )
        {
            // create factory for the random number generator
            using RNGFactory = PMacc::random::RNGProvider< simDim, PMacc::random::methods::XorMin >;
            auto rngFactory = new RNGFactory( Environment<simDim>::get().SubGrid().getLocalDomain().size );

            // init and share random number generator
            PMacc::GridController<simDim>& gridCon = PMacc::Environment<simDim>::get().GridController();
            rngFactory->init( gridCon.getScalarPosition() );
            dc.share( std::shared_ptr< ISimulationData >( rngFactory ) );
        }

        // Initialize synchrotron functions, if there are synchrotron photon species
        if(!bmpl::empty<AllSynchrotronPhotonsSpecies>::value)
        {
            this->synchrotronFunctions.init();
        }

        // Initialize bremsstrahlung lookup tables, if there are species containing bremsstrahlung photons
        if(!bmpl::empty<AllBremsstrahlungPhotonsSpecies>::value)
        {
            ForEach<
                AllBremsstrahlungPhotonsSpecies,
                particles::bremsstrahlung::FillScaledSpectrumMap< bmpl::_1 >
            > fillScaledSpectrumMap;
            fillScaledSpectrumMap(forward(this->scaledBremsstrahlungSpectrumMap));

            this->bremsstrahlungPhotonAngle.init();
        }

        /* Create an empty allocator. This one is resized after all exchanges
         * for particles are created */
        deviceHeap.reset(new DeviceHeap(0));

        ForEach< VectorAllSpecies, particles::CreateSpecies<bmpl::_1> > createSpeciesMemory;
        createSpeciesMemory( deviceHeap, cellDescription );

        size_t freeGpuMem(0);
        Environment<>::get().MemoryInfo().getMemoryInfo(&freeGpuMem);
        if(freeGpuMem < reservedGpuMemorySize)
        {
            PMacc::log< picLog::MEMORY > ("%1% MiB free memory < %2% MiB required reserved memory")
                % (freeGpuMem / 1024 / 1024) % (reservedGpuMemorySize / 1024 / 1024) ;
            std::stringstream msg;
            msg << "Cannot reserve "
                << (reservedGpuMemorySize / 1024 / 1024) << " MiB as there is only "
                << (freeGpuMem / 1024 / 1024) << " MiB free GPU memory left";
            throw std::runtime_error(msg.str());
        }

        size_t heapSize = freeGpuMem - reservedGpuMemorySize;

        if( Environment<>::get().MemoryInfo().isSharedMemoryPool() )
        {
            heapSize /= 2;
            log<picLog::MEMORY > ("Shared RAM between GPU and host detected - using only half of the 'device' memory.");
        }
        else
            log<picLog::MEMORY > ("RAM is NOT shared between GPU and host.");

        // initializing the heap for particles
        deviceHeap->destructiveResize(heapSize);
        MallocMCBuffer<DeviceHeap>* mallocMCBuffer = new MallocMCBuffer<DeviceHeap>(deviceHeap);
        dc.share( std::shared_ptr< ISimulationData >( mallocMCBuffer ) );

        ForEach< VectorAllSpecies, particles::CallCreateParticleBuffer<bmpl::_1> > createParticleBuffer;
        createParticleBuffer( deviceHeap );

        Environment<>::get().MemoryInfo().getMemoryInfo(&freeGpuMem);
        log<picLog::MEMORY > ("free mem after all mem is allocated %1% MiB") % (freeGpuMem / 1024 / 1024);

        IdProvider<simDim>::init();

        fieldB->init( *laser );
        fieldE->init( *laser );
        fieldJ->init( );
        for( uint32_t slot = 0; slot < fieldTmpNumSlots; ++slot)
            fieldTmp.at( slot )->init();

        // create field solver
        this->myFieldSolver = new fieldSolver::FieldSolver(*cellDescription);

        // create current interpolation
        this->myCurrentInterpolation = new fieldSolver::CurrentInterpolation;


        ForEach< VectorAllSpecies, particles::CallInit<bmpl::_1> > particleInit;
        particleInit( );


        /* add CUDA streams to the StreamController for concurrent execution */
        Environment<>::get().StreamController().addStreams(6);
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
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        gc.setStateAfterSlides(0);

        DataConnector &dc = Environment<>::get().DataConnector();
        auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
        auto fieldB = dc.get< FieldB >( FieldB::getName(), true );

        /* fill all objects registed in DataConnector */
        if (initialiserController)
        {
            initialiserController->printInformation();
            if (this->restartRequested)
            {
                /* we do not require --restart-step if a master checkpoint file is found */
                if (this->restartStep < 0)
                {
                    std::vector<uint32_t> checkpoints = readCheckpointMasterFile();

                    if (checkpoints.empty())
                    {
                        throw std::runtime_error(
                                                 "Restart failed. You must provide the '--restart-step' argument. See picongpu --help.");
                    } else
                        this->restartStep = checkpoints.back();
                }

                initialiserController->restart((uint32_t)this->restartStep, this->restartDirectory);
                step = this->restartStep;

                /** restore background fields in GUARD
                 *
                 * loads the outer GUARDS of the global domain for absorbing/open boundary condtions
                 *
                 * @todo as soon as we add GUARD fields to the checkpoint data, e.g. for PML boundary
                 *       conditions, this section needs to be removed
                 */
                cellwiseOperation::CellwiseOperation< GUARD > guardBGField( *cellDescription );
                namespace nvfct = pmacc::nvidia::functors;
                guardBGField( fieldE, nvfct::Add(), FieldBackgroundE( fieldE->getUnit() ),
                              step, FieldBackgroundE::InfluenceParticlePusher );
                guardBGField( fieldB, nvfct::Add(), FieldBackgroundB( fieldB->getUnit() ),
                              step, FieldBackgroundB::InfluenceParticlePusher );

            }
            else
            {
                initialiserController->init();
                ForEach< particles::InitPipeline, particles::CallFunctor<bmpl::_1> > initSpecies;
                initSpecies( step );
            }
        }

        size_t freeGpuMem(0u);
        Environment<>::get().MemoryInfo().getMemoryInfo(&freeGpuMem);
        log<picLog::MEMORY > ("free mem after all particles are initialized %1% MiB") % (freeGpuMem / 1024 / 1024);

        // generate valid GUARDS (overwrite)

        EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldE);
        EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldB);

        dc.releaseData( FieldE::getName() );
        dc.releaseData( FieldB::getName() );

        return step;
    }

    /**
     * Run one simulation step.
     *
     * @param currentStep iteration number of the current step
     */
    virtual void runOneStep(uint32_t currentStep)
    {
        namespace nvfct = PMacc::nvidia::functors;

        typedef typename PMacc::particles::traits::FilterByIdentifier
        <
            VectorAllSpecies,
            momentumPrev1
        >::type VectorSpeciesWithMementumPrev1;

        /* copy attribute momentum to momentumPrev1 */
        ForEach<
            VectorSpeciesWithMementumPrev1,
            particles::Manipulate<
                particles::manipulators::CopyAttribute<
                    momentumPrev1,
                    momentum
                >,
                bmpl::_1
            >
        > copyMomentumPrev1;
        copyMomentumPrev1( currentStep );

        DataConnector &dc = Environment<>::get().DataConnector();

        /* Initialize ionization routine for each species with the flag `ionizers<>` */
        using VectorSpeciesWithIonizers = typename PMacc::particles::traits::FilterByFlag<
            VectorAllSpecies,
            ionizers<>
        >::type;
        ForEach< VectorSpeciesWithIonizers, particles::CallIonization< bmpl::_1 > > particleIonization;
        particleIonization( cellDescription, currentStep );

        /* call the synchrotron radiation module for each radiating species (normally electrons) */
        typedef typename PMacc::particles::traits::FilterByFlag<VectorAllSpecies,
                                                                synchrotronPhotons<> >::type AllSynchrotronPhotonsSpecies;

        ForEach<
            AllSynchrotronPhotonsSpecies,
            particles::CallSynchrotronPhotons< bmpl::_1 >
        > synchrotronRadiation;
        synchrotronRadiation( cellDescription, currentStep, this->synchrotronFunctions );

        /* Bremsstrahlung */
        typedef typename PMacc::particles::traits::FilterByFlag
        <
            VectorAllSpecies,
            bremsstrahlungIons<>
        >::type VectorSpeciesWithBremsstrahlung;
        ForEach<
            VectorSpeciesWithBremsstrahlung,
            particles::CallBremsstrahlung< bmpl::_1 >
        > particleBremsstrahlung;
        particleBremsstrahlung(
            cellDescription,
            currentStep,
            this->scaledBremsstrahlungSpectrumMap,
            this->bremsstrahlungPhotonAngle);

        EventTask initEvent = __getTransactionEvent();
        EventTask updateEvent;
        EventTask commEvent;

        /* push all species */
        particles::PushAllSpecies pushAllSpecies;
        pushAllSpecies( currentStep, initEvent, updateEvent, commEvent );

        __setTransactionEvent(updateEvent);
        /** remove background field for particle pusher */
        auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
        auto fieldB = dc.get< FieldB >( FieldB::getName(), true );
        (*pushBGField)(fieldE, nvfct::Sub(), FieldBackgroundE(fieldE->getUnit()),
                       currentStep, FieldBackgroundE::InfluenceParticlePusher);
        (*pushBGField)(fieldB, nvfct::Sub(), FieldBackgroundB(fieldB->getUnit()),
                       currentStep, FieldBackgroundB::InfluenceParticlePusher);
        dc.releaseData( FieldE::getName() );
        dc.releaseData( FieldB::getName() );

        this->myFieldSolver->update_beforeCurrent(currentStep);

        auto fieldJ = dc.get< FieldJ >( FieldJ::getName(), true );
        FieldJ::ValueType zeroJ( FieldJ::ValueType::create(0.) );
        fieldJ->assign( zeroJ );

        __setTransactionEvent(commEvent);
        (*currentBGField)(fieldJ, nvfct::Add(), FieldBackgroundJ(fieldJ->getUnit()),
                          currentStep, FieldBackgroundJ::activated);
#if (ENABLE_CURRENT == 1)
        typedef typename PMacc::particles::traits::FilterByFlag
        <
            VectorAllSpecies,
            current<>
        >::type VectorSpeciesWithCurrentSolver;
        ForEach<
            VectorSpeciesWithCurrentSolver,
            ComputeCurrent<
                bmpl::_1,
                bmpl::int_<CORE + BORDER>
            >
        > computeCurrent;
        computeCurrent( currentStep );
#endif

#if  (ENABLE_CURRENT == 1)
        if(bmpl::size<VectorSpeciesWithCurrentSolver>::type::value > 0)
        {
            EventTask eRecvCurrent = fieldJ->asyncCommunication(__getTransactionEvent());

            const DataSpace<simDim> currentRecvLower( GetMargin<fieldSolver::CurrentInterpolation>::LowerMargin( ).toRT( ) );
            const DataSpace<simDim> currentRecvUpper( GetMargin<fieldSolver::CurrentInterpolation>::UpperMargin( ).toRT( ) );

            /* without interpolation, we do not need to access the FieldJ GUARD
             * and can therefor overlap communication of GUARD->(ADD)BORDER & computation of CORE */
            if( currentRecvLower == DataSpace<simDim>::create(0) &&
                currentRecvUpper == DataSpace<simDim>::create(0) )
            {
                fieldJ->addCurrentToEMF<CORE >(*myCurrentInterpolation);
                __setTransactionEvent(eRecvCurrent);
                fieldJ->addCurrentToEMF<BORDER >(*myCurrentInterpolation);
            } else
            {
                /* in case we perform a current interpolation/filter, we need
                 * to access the BORDER area from the CORE (and the GUARD area
                 * from the BORDER)
                 * `fieldJ->asyncCommunication` first adds the neighbors' values
                 * to BORDER (send) and then updates the GUARD (receive)
                 * \todo split the last `receive` part in a separate method to
                 *       allow already a computation of CORE */
                __setTransactionEvent(eRecvCurrent);
                fieldJ->addCurrentToEMF<CORE + BORDER>(*myCurrentInterpolation);
            }
        }
#endif
        dc.releaseData( FieldJ::getName() );

        this->myFieldSolver->update_afterCurrent(currentStep);
    }

    virtual void movingWindowCheck(uint32_t currentStep)
    {
        if (MovingWindow::getInstance().slideInCurrentStep(currentStep))
        {
            slide(currentStep);
        }

        /* do not double-add background field on restarts
         * (contained in checkpoint data)
         */
        bool addBgFields = true;
        if( this->restartRequested )
        {
            if( this->restartStep == int32_t(currentStep) )
                addBgFields = false;
        }
        if( addBgFields )
        {
            /** add background field: the movingWindowCheck is just at the start
             * of a time step before all the plugins are called (and the step
             * itself is performed for this time step).
             * Hence the background field is visible for all plugins
             * in between the time steps.
             */
            namespace nvfct = PMacc::nvidia::functors;

            DataConnector &dc = Environment<>::get().DataConnector();

            auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
            auto fieldB = dc.get< FieldB >( FieldB::getName(), true );

            (*pushBGField)( fieldE, nvfct::Add(), FieldBackgroundE(fieldE->getUnit()),
                            currentStep, FieldBackgroundE::InfluenceParticlePusher );
            (*pushBGField)( fieldB, nvfct::Add(), FieldBackgroundB(fieldB->getUnit()),
                            currentStep, FieldBackgroundB::InfluenceParticlePusher );

            dc.releaseData( FieldE::getName() );
            dc.releaseData( FieldB::getName() );
        }
    }

    virtual void resetAll(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
        auto fieldB = dc.get< FieldB >( FieldB::getName(), true );

        fieldB->reset(currentStep);
        fieldE->reset(currentStep);
        ForEach< VectorAllSpecies, particles::CallReset< bmpl::_1 > > callReset;
        callReset( currentStep );

        dc.releaseData( FieldE::getName() );
        dc.releaseData( FieldB::getName() );
    }

    void slide(uint32_t currentStep)
    {
        GridController<simDim>& gc = Environment<simDim>::get().GridController();

        if (gc.slide())
        {
            log<picLog::SIMULATION_STATE > ("slide in step %1%") % currentStep;
            resetAll(currentStep);
            initialiserController->slide(currentStep);
            ForEach< particles::InitPipeline, particles::CallFunctor< bmpl::_1 > > initSpecies;
            initSpecies( currentStep );
        }
    }

    virtual void setInitController(IInitPlugin *initController)
    {

        PMACC_ASSERT(initController != nullptr);
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
            // global size must be a devisor of supercell size
            // note: this is redundant, while using the local condition below
            PMACC_VERIFY(globalGridSize[i] % MappingDesc::SuperCellSize::toRT()[i] == 0);
            // local size must be a devisor of supercell size
            PMACC_VERIFY(gridSizeLocal[i] % MappingDesc::SuperCellSize::toRT()[i] == 0);
            // local size must be at least 3 supercells (1x core + 2x border)
            // note: size of border = guard_size (in supercells)
            // \todo we have to add the guard_x/y/z for modified supercells here
            PMACC_VERIFY((uint32_t) gridSizeLocal[i] / MappingDesc::SuperCellSize::toRT()[i] >= 3 * GUARD_SIZE);
        }
    }

protected:
    std::shared_ptr<DeviceHeap> deviceHeap;

    // field solver
    fieldSolver::FieldSolver* myFieldSolver;
    fieldSolver::CurrentInterpolation* myCurrentInterpolation;

    cellwiseOperation::CellwiseOperation< CORE + BORDER + GUARD >* pushBGField;
    cellwiseOperation::CellwiseOperation< CORE + BORDER + GUARD >* currentBGField;

    LaserPhysics *laser;

    // creates lookup tables for the bremsstrahlung effect
    // map<atomic number, scaled bremsstrahlung spectrum>
    std::map<float_X, particles::bremsstrahlung::ScaledSpectrum> scaledBremsstrahlungSpectrumMap;
    particles::bremsstrahlung::GetPhotonAngle bremsstrahlungPhotonAngle;

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
    DataSpace<simDim> gridOffset;
    std::vector<uint32_t> periodic;

    std::vector<std::string> gridDistribution;

    bool slidingWindow;
};
} /* namespace picongpu */

#include "fields/Fields.tpp"
#include "particles/synchrotronPhotons/SynchrotronFunctions.tpp"
#include "particles/bremsstrahlung/Bremsstrahlung.tpp"
#include "particles/bremsstrahlung/ScaledSpectrum.tpp"
