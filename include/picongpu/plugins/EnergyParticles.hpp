/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch, Benjamin Worpitz
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

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>


namespace picongpu
{
    /** accumulate the kinetic and total energy
     *
     * All energies are summed over all particles of a species.
     */
    struct KernelEnergyParticles
    {
        /** accumulate particle energies
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_DBox pmacc::DataBox, type of the memory box for the reduced energies
         * @tparam T_Mapping mapper functor type
         *
         * @param pb particle memory
         * @param gEnergy storage for the reduced energies
         *                (two elements 0 == kinetic; 1 == total energy)
         * @param mapper functor to map a block to a supercell
         */
        template<typename T_ParBox, typename T_DBox, typename T_Mapping, typename T_Worker, typename T_Filter>
        DINLINE void operator()(T_Worker const& worker, T_ParBox pb, T_DBox gEnergy, T_Mapping mapper, T_Filter filter)
            const
        {
            // shared kinetic energy
            PMACC_SMEM(worker, shEnergyKin, float_X);
            // shared total energy
            PMACC_SMEM(worker, shEnergy, float_X);

            // sum kinetic energy for all particles touched by the virtual thread
            float_X localEnergyKin(0.0);
            float_X localEnergy(0.0);

            auto masterOnly = lockstep::makeMaster(worker);

            masterOnly(
                [&]()
                {
                    // set shared kinetic energy to zero
                    shEnergyKin = float_X(0.0);
                    // set shared total energy to zero
                    shEnergy = float_X(0.0);
                });

            worker.sync();

            DataSpace<simDim> const superCellIdx(
                mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdx);

            // end kernel if we have no particles
            if(!forEachParticle.hasParticles())
                return;

            auto accFilter = filter(worker, superCellIdx - mapper.getGuardingSuperCells());

            forEachParticle(
                [&accFilter, &localEnergyKin, &localEnergy](auto const& lockstepWorker, auto& particle)
                {
                    if(accFilter(lockstepWorker, particle))
                    {
                        float3_X const mom = particle[momentum_];
                        // compute square of absolute momentum of the particle
                        float_X const mom2 = pmacc::math::l2norm2(mom);
                        float_X const weighting = particle[weighting_];
                        float_X const mass = attribute::getMass(weighting, particle);
                        float_X const c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

                        // calculate kinetic energy of the macro particle
                        localEnergyKin += KinEnergy<>()(mom, mass);

                        /* total energy for particles:
                         *    E^2 = p^2*c^2 + m^2*c^4
                         *        = c^2 * [p^2 + m^2*c^2]
                         */
                        localEnergy += math::sqrt(mom2 + mass * mass * c2) * SPEED_OF_LIGHT;
                    }
                });

            // each virtual thread adds the energies to the shared memory
            cupla::atomicAdd(worker.getAcc(), &shEnergyKin, localEnergyKin, ::alpaka::hierarchy::Threads{});
            cupla::atomicAdd(worker.getAcc(), &shEnergy, localEnergy, ::alpaka::hierarchy::Threads{});

            // wait that all virtual threads updated the shared memory energies
            worker.sync();

            // add energies on global level using global memory
            masterOnly(

                [&]()
                {
                    // add kinetic energy
                    cupla::atomicAdd(
                        worker.getAcc(),
                        &(gEnergy[0]),
                        static_cast<float_64>(shEnergyKin),
                        ::alpaka::hierarchy::Blocks{});
                    // add total energy
                    cupla::atomicAdd(
                        worker.getAcc(),
                        &(gEnergy[1]),
                        static_cast<float_64>(shEnergy),
                        ::alpaka::hierarchy::Blocks{});
                });
        }
    };

    template<typename ParticlesType>
    class EnergyParticles : public plugins::multi::IInstance
    {
    public:
        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::shared_ptr<IInstance>(new EnergyParticles<ParticlesType>(help, id, cellDescription));
            }

            // find all valid filter for the current used species
            template<typename T>
            using Op = typename particles::traits::GenerateSolversIfSpeciesEligible<T, ParticlesType>::type;
            using EligibleFilters = pmacc::mp_flatten<pmacc::mp_transform<Op, particles::filter::AllParticleFilters>>;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod
                = {"period",
                   "compute kinetic and total energy [for each n-th step] enable plugin by setting a non-zero value"};
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

                // check if user passed filter name are valid
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

            std::string const name = "EnergyParticles";
            //! short description of the plugin
            std::string const description = "calculate the energy of a species";
            //! prefix used for command line arguments
            std::string const prefix = ParticlesType::FrameType::getName() + std::string("_energy");
        };

        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        EnergyParticles(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            filename = m_help->getOptionPrefix() + "_" + m_help->filter.get(m_id) + ".dat";

            // decide which MPI-rank writes output
            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());

            // create two ints on gpu and host
            gEnergy = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(2));

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

                // create header of the file
                outFile << "#step Ekin_Joule E_Joule"
                        << " \n";
            }

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }

        ~EnergyParticles() override
        {
            if(writeToFile)
            {
                outFile.flush();
                // flush cached data to file
                outFile << std::endl;

                if(outFile.fail())
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close();
            }
        }

        /** this code is executed if the current time step is supposed to compute
         * the energy
         */
        void notify(uint32_t currentStep) override
        {
            // call the method that calls the plugin kernel
            calculateEnergyParticles<CORE + BORDER>(currentStep);
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
        void calculateEnergyParticles(uint32_t currentStep)
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            // use data connector to get particle data
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            // initialize global energies with zero
            gEnergy->getDeviceBuffer().setValue(0.0);

            auto const mapper = makeAreaMapper<AREA>(*m_cellDescription);

            auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();
            auto kernel = PMACC_LOCKSTEP_KERNEL(KernelEnergyParticles{}, workerCfg)(mapper.getGridDim());
            auto binaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                gEnergy->getDeviceBuffer().getDataBox(),
                mapper,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                binaryKernel);

            // get energy from GPU
            gEnergy->deviceToHost();

            // create storage for the global reduced result
            float_64 reducedEnergy[2];

            // add energies from all GPUs using MPI
            reduce(
                pmacc::math::operation::Add(),
                reducedEnergy,
                gEnergy->getHostBuffer().getBasePointer(),
                2,
                mpi::reduceMethods::Reduce());

            /* print timestep, kinetic energy and total energy to file: */
            if(writeToFile)
            {
                using dbl = std::numeric_limits<float_64>;

                outFile.precision(dbl::digits10);
                outFile << currentStep << " " << std::scientific << reducedEnergy[0] * UNIT_ENERGY << " "
                        << reducedEnergy[1] * UNIT_ENERGY << std::endl;
            }
        }

        //! energy values (global on GPU)
        std::unique_ptr<GridBuffer<float_64, DIM1>> gEnergy;

        MappingDesc* m_cellDescription;

        //! output file name
        std::string filename;

        //! file output stream
        std::ofstream outFile;

        /** only one MPI rank creates a file
         *
         * true if this MPI rank creates the file, else false
         */
        bool writeToFile = false;

        //! MPI reduce to add all energies over several GPUs
        mpi::MPIReduce reduce;

        std::shared_ptr<Help> m_help;
        size_t m_id;
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, EnergyParticles<T_UnspecifiedSpecies>>
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
