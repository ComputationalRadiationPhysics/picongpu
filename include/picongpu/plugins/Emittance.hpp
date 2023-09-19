/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch, Benjamin Worpitz,
 *                     Sophie Rudat
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/particles/traits/GenerateSolversIfSpeciesEligible.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/common/txtFileHandling.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/multi.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/kernel/atomic.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace picongpu
{
    //! calculates the emittance in x direction along the y axis
    struct KernelCalcEmittance
    {
        /** calculates the sum of x^2, ux^2 and x*ux and counts electrons
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_DBox pmacc::DataBox, type of the memory box for the reduced values
         * @tparam T_Mapping mapper functor type
         * @tparam T_Worker lockstep worker type
         * @tparam T_Filter particle filter functor type
         *
         * @param pb particle memory
         * @param gSumMom2 global (reduced) sum of momentum square
         * @param gSumPos2 global (reduced) sum of position square
         * @param gSumMomPos global (reduced) sum of momentum times position
         * @param gCount_e global real particle counter
         * @param mapper functor to map a block to a supercell
         */
        template<typename T_ParBox, typename T_DBox, typename T_Mapping, typename T_Worker, typename T_Filter>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox pb,
            T_DBox gSumMom2,
            T_DBox gSumPos2,
            T_DBox gSumMomPos,
            T_DBox gCount_e,
            DataSpace<simDim> globalOffset,
            const int subGridY,
            T_Mapping mapper,
            T_Filter filter) const
        {
            // shared sums of x^2, ux^2, x*ux, particle counter
            PMACC_SMEM(worker, shSumMom2, memory::Array<float_X, SuperCellSize::y::value>);
            PMACC_SMEM(worker, shSumPos2, memory::Array<float_X, SuperCellSize::y::value>);
            PMACC_SMEM(worker, shSumMomPos, memory::Array<float_X, SuperCellSize::y::value>);
            PMACC_SMEM(worker, shCount_e, memory::Array<float_X, SuperCellSize::y::value>);

            DataSpace<simDim> const superCellIdx(
                mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdx);

            // end kernel if we have no particles
            if(!forEachParticle.hasParticles())
                return;

            auto forEachSuperCellInY = lockstep::makeForEach<SuperCellSize::y::value>(worker);

            forEachSuperCellInY(
                [&](uint32_t const linearIdx)
                {
                    // set shared sums of x^2, ux^2, x*ux, particle counter to zero
                    shSumMom2[linearIdx] = 0.0_X;
                    shSumPos2[linearIdx] = 0.0_X;
                    shSumMomPos[linearIdx] = 0.0_X;
                    shCount_e[linearIdx] = 0.0_X;
                });

            worker.sync();

            auto accFilter = filter(worker, superCellIdx - mapper.getGuardingSuperCells());

            forEachParticle(
                [&accFilter, &mapper, &globalOffset, &superCellIdx, &shCount_e, &shSumMom2, &shSumPos2, &shSumMomPos](
                    auto const& lockstepWorker,
                    auto& particle)
                {
                    if(accFilter(lockstepWorker, particle))
                    {
                        float_X const weighting = particle[weighting_];
                        float_X const normedWeighting
                            = weighting / float_X(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);
                        float3_X const mom = particle[momentum_] / weighting;
                        floatD_X const pos = particle[position_];
                        lcellId_t const cellIdx = particle[localCellIdx_];
                        DataSpace<simDim> const frameCellOffset(
                            DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize>(cellIdx));
                        auto const localSupercellStart
                            = (superCellIdx - mapper.getGuardingSuperCells()) * MappingDesc::SuperCellSize::toRT();
                        int const index_y = frameCellOffset.y();
                        auto const globalCellOffset = globalOffset + localSupercellStart + frameCellOffset;
                        float_X const posX = (float_X(globalCellOffset.x()) + pos.x()) * cellSize.x();

                        cupla::atomicAdd(
                            lockstepWorker.getAcc(),
                            &(shCount_e[index_y]),
                            normedWeighting,
                            ::alpaka::hierarchy::Threads{});
                        // weighted sum of single Electron values (Momentum = particle_momentum/weighting)
                        cupla::atomicAdd(
                            lockstepWorker.getAcc(),
                            &(shSumMom2[index_y]),
                            mom.x() * mom.x() * normedWeighting,
                            ::alpaka::hierarchy::Threads{});
                        cupla::atomicAdd(
                            lockstepWorker.getAcc(),
                            &(shSumPos2[index_y]),
                            posX * posX * normedWeighting,
                            ::alpaka::hierarchy::Threads{});
                        cupla::atomicAdd(
                            lockstepWorker.getAcc(),
                            &(shSumMomPos[index_y]),
                            mom.x() * posX * normedWeighting,
                            ::alpaka::hierarchy::Threads{});
                    }
                });

            // wait that all virtual threads updated the shared memory
            worker.sync();

            const int gOffset
                = ((superCellIdx - mapper.getGuardingSuperCells()) * MappingDesc::SuperCellSize::toRT()).y();

            forEachSuperCellInY(
                [&](uint32_t const linearIdx)
                {
                    cupla::atomicAdd(
                        worker.getAcc(),
                        &(gSumMom2[gOffset + linearIdx]),
                        static_cast<float_64>(shSumMom2[linearIdx]),
                        ::alpaka::hierarchy::Blocks{});
                    cupla::atomicAdd(
                        worker.getAcc(),
                        &(gSumPos2[gOffset + linearIdx]),
                        static_cast<float_64>(shSumPos2[linearIdx]),
                        ::alpaka::hierarchy::Blocks{});
                    cupla::atomicAdd(

                        worker.getAcc(),
                        &(gSumMomPos[gOffset + linearIdx]),
                        static_cast<float_64>(shSumMomPos[linearIdx]),
                        ::alpaka::hierarchy::Blocks{});
                    cupla::atomicAdd(
                        worker.getAcc(),
                        &(gCount_e[gOffset + linearIdx]),
                        static_cast<float_64>(shCount_e[linearIdx]),
                        ::alpaka::hierarchy::Blocks{});
                });
        }
    };


    template<typename ParticlesType>
    class CalcEmittance : public plugins::multi::IInstance
    {
    public:
        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [ 0;help->getNumPlugins( ) )
             */
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::shared_ptr<IInstance>(new CalcEmittance<ParticlesType>(help, id, cellDescription));
            }

            // find all valid filter for the current used species
            template<typename T>
            using Op = typename particles::traits::GenerateSolversIfSpeciesEligible<T, ParticlesType>::type;
            using EligibleFilters = pmacc::mp_flatten<pmacc::mp_transform<Op, particles::filter::AllParticleFilters>>;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod
                = {"period", "compute slice emittance[for each n-th step] enable plugin by setting a non-zero value"};
            plugins::multi::Option<std::string> filter = {"filter", "particle filter: "};

            //! string list with all possible particle filters
            std::string concatenatedFilterNames;
            std::vector<std::string> allowedFilters;

            ///! method used by plugin controller to get --help description
            void registerHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                meta::ForEach<EligibleFilters, plugins::misc::AppendName<boost::mpl::_1>> getEligibleFilterNames;
                getEligibleFilterNames(allowedFilters);

                concatenatedFilterNames = plugins::misc::concatenateToString(allowedFilters, ", ");

                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                filter.registerHelp(desc, masterPrefix + prefix, std::string("[") + concatenatedFilterNames + "]");
            }

            void expandHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
            }


            void validateOptions() override
            {
                if(notifyPeriod.size() != filter.size())
                    throw std::runtime_error(
                        name + ": parameter filter and period are not used the same number of times");

                // check if user passed filter name is valid
                for(auto const& filterName : filter)
                {
                    if(std::find(allowedFilters.begin(), allowedFilters.end(), filterName) == allowedFilters.end())
                    {
                        throw std::runtime_error(name + ": unknown filter '" + filterName + "'");
                    }
                }
            }

            size_t getNumPlugins() const override
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const override
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const override
            {
                return name;
            }

            std::string const name = "CalcEmittance";
            //! short description of the plugin
            std::string const description = "calculate the slice emittance of a species";
            //! prefix used for command line arguments
            std::string const prefix = ParticlesType::FrameType::getName() + std::string("_emittance");
        };

        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        CalcEmittance(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            filename = m_help->getOptionPrefix() + "_" + m_help->filter.get(m_id) + ".dat";

            // reduce in same x-z plane
            constexpr uint32_t r_element = 1u; // y-direction

            /* reduce-add particle properties from other GPUs in same plane: range [r;r+dr]
             * to "lowest" node in range
             * e.g.: slice emittance in space y: reduce-add all nodes with same y range in
             *                         spatial x and z direction to node with
             *                         lowest x and z position ("corner") and same x range
             */
            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
            pmacc::math::Size_t<simDim> gpuDim = gc.getGpuNodes();
            pmacc::math::Int<simDim> gpuPos = gc.getPosition();

            /* my plane means: the r_element I am calculating should be 1GPU in width */
            pmacc::math::Size_t<simDim> sizeTransversalPlane(gpuDim);
            sizeTransversalPlane[r_element] = 1;

            // avoid deadlock for following, blocking MPI operations
            eventSystem::getTransactionEvent().waitForFinished();


            for(int planePos = 0; planePos <= (int) gpuDim[r_element]; ++planePos)
            {
                auto mpiReduce = std::make_unique<mpi::MPIReduce>();
                bool isInGroup = (gpuPos[r_element] == planePos);

                mpiReduce->participate(isInGroup);
                if(isInGroup)
                {
                    this->isPlaneReduceRoot = mpiReduce->hasResult(::pmacc::mpi::reduceMethods::Reduce{});
                    this->planeReduce = std::move(mpiReduce);
                }
            }

            /* Create MPI communicator for openPMD IO with ranks of each plane reduce root */
            {
                /* Array with root ranks of the planeReduce operations */
                std::vector<int> planeReduceRootRanks(gc.getGlobalSize(), -1);
                /* Am I one of the planeReduce root ranks? my global rank : -1 */
                int myRootRank = gc.getGlobalRank() * isPlaneReduceRoot - (!isPlaneReduceRoot);

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_Group world_group, new_group;
                MPI_CHECK(MPI_Allgather(
                    &myRootRank,
                    1,
                    MPI_INT,
                    planeReduceRootRanks.data(),
                    1,
                    MPI_INT,
                    gc.getCommunicator().getMPIComm()));

                /* remove all non-roots (-1 values) */
                std::sort(planeReduceRootRanks.begin(), planeReduceRootRanks.end());
                std::vector<int> ranks(
                    std::lower_bound(planeReduceRootRanks.begin(), planeReduceRootRanks.end(), 0),
                    planeReduceRootRanks.end());

                MPI_CHECK(MPI_Comm_group(gc.getCommunicator().getMPIComm(), &world_group));
                MPI_CHECK(MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group));
                MPI_CHECK(MPI_Comm_create(gc.getCommunicator().getMPIComm(), new_group, &commGather));
                MPI_CHECK(MPI_Group_free(&new_group));
                MPI_CHECK(MPI_Group_free(&world_group));
            }

            // decide which MPI-rank writes output
            int gatherRank = -1;
            if(commGather != MPI_COMM_NULL)
                MPI_CHECK(MPI_Comm_rank(commGather, &gatherRank));
            writeToFile = (gatherRank == 0);

            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            gSumMom2
                = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(subGrid.getLocalDomain().size.y()));
            gSumPos2
                = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(subGrid.getLocalDomain().size.y()));
            gSumMomPos
                = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(subGrid.getLocalDomain().size.y()));
            gCount_e
                = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(subGrid.getLocalDomain().size.y()));

            // only MPI rank that writes to file
            if(writeToFile)
            {
                // open output file
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);

                // error handling
                if(!outFile)
                {
                    std::cerr << "Can't open file [" << filename << "] for output, diasble plugin output. "
                              << std::endl;
                    writeToFile = false;
                }
            }

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }

        ~CalcEmittance() override
        {
            if(writeToFile)
            {
                // flush cached data to file
                outFile.flush() << std::endl;

                if(outFile.fail())
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close();
            }
        }

        /** this code is executed if the current time step is supposed to compute
         * gSumMom2, gSumPos2, gSumMomPos, gCount_e
         */
        void notify(uint32_t currentStep) override
        {
            // call the method that calls the plugin kernel
            calculateCalcEmittance<CORE + BORDER>(currentStep);
        }


        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
            if(!writeToFile)
                return;

            writeToFile = restoreTxtFile(outFile, filename, restartStep, restartDirectory);
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
            if(!writeToFile)
                return;

            checkpointTxtFile(outFile, filename, currentStep, checkpointDirectory);
        }

    private:
        //! method to call analysis and plugin-kernel calls
        template<uint32_t AREA>
        void calculateCalcEmittance(uint32_t currentStep)
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            // use data connector to get particle data
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            gSumMom2->getDeviceBuffer().setValue(0.0);
            gSumPos2->getDeviceBuffer().setValue(0.0);
            gSumMomPos->getDeviceBuffer().setValue(0.0);
            gCount_e->getDeviceBuffer().setValue(0.0);

            auto const mapper = makeAreaMapper<AREA>(*m_cellDescription);

            auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();
            auto kernel = PMACC_LOCKSTEP_KERNEL(KernelCalcEmittance{}, workerCfg)(mapper.getGridDim());

            // Some variables required so that it is possible for the kernel
            // to calculate the absolute position of the particles
            DataSpace<simDim> localSize(m_cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const int globalDomainSizeY = subGrid.getGlobalDomain().size.y();
            auto movingWindow = MovingWindow::getInstance().getWindow(currentStep);
            DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);

            auto binaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                gSumMom2->getDeviceBuffer().getDataBox(),
                gSumPos2->getDeviceBuffer().getDataBox(),
                gSumMomPos->getDeviceBuffer().getDataBox(),
                gCount_e->getDeviceBuffer().getDataBox(),
                globalOffset,
                globalDomainSizeY,
                mapper,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                binaryKernel);

            // get gSum, ... from GPU
            gSumMom2->deviceToHost();
            gSumPos2->deviceToHost();
            gSumMomPos->deviceToHost();
            gCount_e->deviceToHost();

            auto localDomSizeY = subGrid.getGlobalDomain().size.y();

            pmacc::HostBuffer<float_64, DIM1> reducedSumMom2(localDomSizeY);
            pmacc::HostBuffer<float_64, DIM1> reducedSumPos2(localDomSizeY);
            pmacc::HostBuffer<float_64, DIM1> reducedSumMomPos(localDomSizeY);
            pmacc::HostBuffer<float_64, DIM1> reducedCount_e(localDomSizeY);
            reducedSumMom2.setValue(0.0);
            reducedSumPos2.setValue(0.0);
            reducedSumMomPos.setValue(0.0);
            reducedCount_e.setValue(0.0);

            // add gSum values from all GPUs using MPI
            (*planeReduce)(
                pmacc::math::operation::Add(),
                reducedSumMom2.getBasePointer(),
                gSumMom2->getHostBuffer().getBasePointer(),
                reducedSumMom2.getCurrentSize(),
                mpi::reduceMethods::Reduce());

            (*planeReduce)(
                pmacc::math::operation::Add(),
                reducedSumPos2.getBasePointer(),
                gSumPos2->getHostBuffer().getBasePointer(),
                reducedSumPos2.getCurrentSize(),
                mpi::reduceMethods::Reduce());

            (*planeReduce)(
                pmacc::math::operation::Add(),
                reducedSumMomPos.getBasePointer(),
                gSumMomPos->getHostBuffer().getBasePointer(),
                reducedSumMomPos.getCurrentSize(),
                mpi::reduceMethods::Reduce());

            (*planeReduce)(
                pmacc::math::operation::Add(),
                reducedCount_e.getBasePointer(),
                gCount_e->getHostBuffer().getBasePointer(),
                reducedCount_e.getCurrentSize(),
                mpi::reduceMethods::Reduce());


            /** all non-reduce-root processes are done now */
            if(!isPlaneReduceRoot)
                return;


            // gather to file writer
            pmacc::HostBuffer<float_64, DIM1> globalSumMom2(globalDomainSizeY);
            pmacc::HostBuffer<float_64, DIM1> globalSumPos2(globalDomainSizeY);
            pmacc::HostBuffer<float_64, DIM1> globalSumMomPos(globalDomainSizeY);
            pmacc::HostBuffer<float_64, DIM1> globalCount_e(globalDomainSizeY);

            // gather y offsets, so we can store our gathered data in the right order
            int gatherSize = -1;
            MPI_CHECK(MPI_Comm_size(commGather, &gatherSize));
            std::vector<int> y_offsets(gatherSize);
            std::vector<int> y_sizes(gatherSize);
            long int const y_off = subGrid.getLocalDomain().offset.y();
            int const y_siz = subGrid.getLocalDomain().size.y();

            MPI_CHECK(MPI_Gather(&y_off, 1, MPI_INT, y_offsets.data(), 1, MPI_INT, 0, commGather));
            MPI_CHECK(MPI_Gather(&y_siz, 1, MPI_INT, y_sizes.data(), 1, MPI_INT, 0, commGather));

            int mpiGlobalSizeY = std::accumulate(y_sizes.begin(), y_sizes.end(), 0);

            if(writeToFile)
            {
                PMACC_VERIFY_MSG(
                    mpiGlobalSizeY == globalDomainSizeY,
                    std::string(
                        "Number of elements calculated with MPI_Gather and global domain size in Y-direction must "
                        "be equal. ")
                        + std::to_string(mpiGlobalSizeY) + " != " + std::to_string(globalDomainSizeY));
            }

            MPI_CHECK(MPI_Gatherv(
                reducedSumMom2.getBasePointer(),
                localDomSizeY,
                MPI_DOUBLE,
                globalSumMom2.getBasePointer(),
                y_sizes.data(),
                y_offsets.data(),
                MPI_DOUBLE,
                0,
                commGather));
            MPI_CHECK(MPI_Gatherv(
                reducedSumPos2.getBasePointer(),
                localDomSizeY,
                MPI_DOUBLE,
                globalSumPos2.getBasePointer(),
                y_sizes.data(),
                y_offsets.data(),
                MPI_DOUBLE,
                0,
                commGather));
            MPI_CHECK(MPI_Gatherv(
                reducedSumMomPos.getBasePointer(),
                localDomSizeY,
                MPI_DOUBLE,
                globalSumMomPos.getBasePointer(),
                y_sizes.data(),
                y_offsets.data(),
                MPI_DOUBLE,
                0,
                commGather));
            MPI_CHECK(MPI_Gatherv(
                reducedCount_e.getBasePointer(),
                localDomSizeY,
                MPI_DOUBLE,
                globalCount_e.getBasePointer(),
                y_sizes.data(),
                y_offsets.data(),
                MPI_DOUBLE,
                0,
                commGather));

            /* print timestep, emittance to file: */
            if(writeToFile)
            {
                using dbl = std::numeric_limits<float_64>;
                outFile.precision(dbl::digits10);
                if(currentStep > 0.0)
                {
                    int startWindow_y = movingWindow.globalDimensions.offset.y();
                    int endWindow_y = movingWindow.globalDimensions.size.y() + startWindow_y;
                    if(fisttimestep == true)
                    {
                        outFile << "#step emit_all" << std::scientific;
                        for(int i = startWindow_y; i < (endWindow_y + 10); i += 10)
                        {
                            outFile << " " << i * SI::CELL_HEIGHT_SI;
                        }
                        outFile << std::endl;
                        fisttimestep = false;
                    }
                    outFile << currentStep << " " << std::scientific;

                    long double numElec_all = 0.0;
                    long double ux2_all = 0.0;
                    long double pos2_SI_all = 0.0;
                    long double xux_all = 0.0;

                    for(int i = startWindow_y; i < endWindow_y; i++)
                    {
                        numElec_all += static_cast<long double>(globalCount_e.getBasePointer()[i]);
                        ux2_all += static_cast<long double>(globalSumMom2.getBasePointer()[i]) * UNIT_MASS * UNIT_MASS
                            / (SI::ELECTRON_MASS_SI * SI::ELECTRON_MASS_SI);
                        pos2_SI_all
                            += static_cast<long double>(globalSumPos2.getBasePointer()[i]) * UNIT_LENGTH * UNIT_LENGTH;
                        xux_all += static_cast<long double>(globalSumMomPos.getBasePointer()[i]) * UNIT_MASS
                            * UNIT_LENGTH / SI::ELECTRON_MASS_SI;
                    }
                    /* the scaling with normalized weighting (weighting /
                     * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE) is compendated by the division by
                     * (normalized) number of particles
                     */
                    float_64 emit_all = math::sqrt(
                                            static_cast<float_64>(pos2_SI_all) * static_cast<float_64>(ux2_all)
                                            - static_cast<float_64>(xux_all) * static_cast<float_64>(xux_all))
                        / static_cast<float_64>(numElec_all);

                    if(emit_all > 0.0)
                    {
                        outFile << emit_all << " ";
                    }
                    else
                    {
                        outFile << "0.0 ";
                    }

                    for(int i = startWindow_y; i < endWindow_y; i += 10)
                    {
                        float_64 numElec = globalCount_e.getBasePointer()[i];
                        float_64 mom2_SI
                            = globalSumMom2.getBasePointer()[i] * UNIT_MASS * UNIT_SPEED * UNIT_MASS * UNIT_SPEED;
                        float_64 pos2_SI = globalSumPos2.getBasePointer()[i] * UNIT_LENGTH * UNIT_LENGTH;
                        float_64 mompos_SI
                            = globalSumMomPos.getBasePointer()[i] * UNIT_MASS * UNIT_SPEED * UNIT_LENGTH;
                        for(int j = i + 1; j < i + 10 && j < endWindow_y; j++)
                        {
                            numElec += globalCount_e.getBasePointer()[j];
                            mom2_SI
                                += globalSumMom2.getBasePointer()[j] * UNIT_MASS * UNIT_SPEED * UNIT_MASS * UNIT_SPEED;
                            pos2_SI += globalSumPos2.getBasePointer()[j] * UNIT_LENGTH * UNIT_LENGTH;
                            mompos_SI += globalSumMomPos.getBasePointer()[j] * UNIT_MASS * UNIT_SPEED * UNIT_LENGTH;
                        }
                        float_64 ux2
                            = mom2_SI / (UNIT_SPEED * UNIT_SPEED * SI::ELECTRON_MASS_SI * SI::ELECTRON_MASS_SI);
                        float_64 xux = mompos_SI / (UNIT_SPEED * SI::ELECTRON_MASS_SI);
                        float_64 emit = math::sqrt((pos2_SI * ux2 - xux * xux)) / numElec;
                        if(numElec < std::numeric_limits<float_64>::epsilon())
                        {
                            outFile << "0.0 ";
                        }
                        else if(emit > 0.0 && emit < std::numeric_limits<float_64>::max())
                        {
                            outFile << emit << " ";
                        }
                        else
                        {
                            outFile << "-0.0 ";
                        }
                    }
                    outFile << std::endl;
                }
            }
        }

        std::unique_ptr<GridBuffer<float_64, DIM1>> gSumMom2;

        std::unique_ptr<GridBuffer<float_64, DIM1>> gSumPos2;

        std::unique_ptr<GridBuffer<float_64, DIM1>> gSumMomPos;

        std::unique_ptr<GridBuffer<float_64, DIM1>> gCount_e;

        MappingDesc* m_cellDescription = nullptr;

        //! output file name
        std::string filename;

        //! file output stream
        std::ofstream outFile;

        /** only one MPI rank creates a file
         *
         * true if this MPI rank creates the file, else false
         */
        bool writeToFile = false;
        bool fisttimestep = true;

        /** reduce functor to a single host per plane */
        std::unique_ptr<pmacc::mpi::MPIReduce> planeReduce;
        bool isPlaneReduceRoot = false;

        /** MPI communicator that contains the root ranks of the \p planeReduce
         */
        MPI_Comm commGather = MPI_COMM_NULL;

        std::shared_ptr<Help> m_help;
        size_t m_id;
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, CalcEmittance<T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                // this plugin needs at least the weighting and momentum attributes
                using RequiredIdentifiers = MakeSeq_t<weighting, momentum>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                // and also a mass ratio for energy calculation from momentum
                using SpeciesHasFlags = typename pmacc::traits::HasFlag<FrameType, massRatio<>>::type;

                using type = pmacc::mp_and<SpeciesHasIdentifiers, SpeciesHasFlags>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
