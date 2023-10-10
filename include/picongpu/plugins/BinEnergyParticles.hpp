/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch
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

#include "common/txtFileHandling.hpp"
#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/particles/traits/GenerateSolversIfSpeciesEligible.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/multi.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace picongpu
{
    using namespace pmacc;

    namespace po = boost::program_options;

    /** calculate a energy histogram of a species
     *
     * if a particle filter is given, only the filtered particles are counted
     */
    struct KernelBinEnergyParticles
    {
        /* sum up the energy of all particles
         *
         * the kinetic energy of all active particles will be calculated
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_BinBox pmacc::DataBox, box type for the histogram in global memory
         * @tparam T_Mapping type of the mapper to map a cupla block to a supercell index
         * @tparam T_Worker lockstep worker type
         *
         * @param acc alpaka accelerator
         * @param pb box with access to the particles of the current used species
         * @param gBins box with memory for resulting histogram
         * @param numBins number of bins in the histogram (must be fit into the shared memory)
         * @param minEnergy particle energy for the first bin
         * @param maxEnergy particle energy for the last bin
         * @param mapper functor to map a cupla block to a supercells index
         */
        template<typename T_ParBox, typename T_BinBox, typename T_Mapping, typename T_Filter, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox pb,
            T_BinBox gBins,
            int const numBins,
            float_X const minEnergy,
            float_X const maxEnergy,
            T_Mapping const mapper,
            T_Filter filter) const
        {
            constexpr uint32_t numWorkers = T_Worker::getNumWorkers();

            /* shBins index can go from 0 to (numBins+2)-1
             * 0 is for <minEnergy
             * (numBins+2)-1 is for >maxEnergy
             */
            /* size must be numBins+2 because we have <min and >max */
            float_X* shBin = ::alpaka::getDynSharedMem<float_X>(worker.getAcc());

            int const realNumBins = numBins + 2;

            DataSpace<simDim> const superCellIdx(
                mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdx);

            // end kernel if we have no particles
            if(!forEachParticle.hasParticles())
                return;

            lockstep::makeForEach<numWorkers>(worker)(
                [&](uint32_t const linearIdx)
                {
                    /* set all bins to 0 */
                    for(int i = linearIdx; i < realNumBins; i += numWorkers)
                        shBin[i] = float_X(0.);
                });

            worker.sync();

            auto accFilter = filter(worker, superCellIdx - mapper.getGuardingSuperCells());

            forEachParticle(
                [&accFilter, &shBin, minEnergy, maxEnergy, numBins](auto const& lockstepWorker, auto& particle)
                {
                    if(accFilter(lockstepWorker, particle))
                    {
                        /* kinetic Energy for Particles: E^2 = p^2*c^2 + m^2*c^4
                         *                                   = c^2 * [p^2 + m^2*c^2]
                         */
                        float3_X const mom = particle[momentum_];
                        float_X const weighting = particle[weighting_];
                        float_X const mass = attribute::getMass(weighting, particle);

                        // calculate kinetic energy of the macro particle
                        float_X localEnergy = KinEnergy<>()(mom, mass);

                        localEnergy /= weighting;

                        /* +1 move value from 1 to numBins+1 */
                        int binNumber
                            = math::floor(
                                  (localEnergy - minEnergy) / (maxEnergy - minEnergy) * static_cast<float_X>(numBins))
                            + 1;

                        int const maxBin = numBins + 1;

                        /* all entries larger than maxEnergy go into bin maxBin */
                        binNumber = binNumber < maxBin ? binNumber : maxBin;

                        /* all entries smaller than minEnergy go into bin zero */
                        binNumber = binNumber > 0 ? binNumber : 0;

                        /*!\todo: we can't use 64bit type on this place (NVIDIA BUG?)
                         * COMPILER ERROR: ptxas /tmp/tmpxft_00005da6_00000000-2_main.ptx, line 4246; error   :
                         * Global state space expected for instruction 'atom' I think this is a problem with
                         * extern shared mem and atmic (only on TESLA) NEXT BUG: don't do uint32_t
                         * w=__float2uint_rn(weighting); and use w for atomic, this create wrong results
                         *
                         * uses a normed float weighting to avoid an overflow of the floating point result
                         * for the reduced weighting if the particle weighting is very large
                         */
                        float_X const normedWeighting
                            = weighting / float_X(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);
                        cupla::atomicAdd(
                            lockstepWorker.getAcc(),
                            &(shBin[binNumber]),
                            normedWeighting,
                            ::alpaka::hierarchy::Threads{});
                    }
                });

            worker.sync();

            lockstep::makeForEach<numWorkers>(worker)(
                [&](uint32_t const linearIdx)
                {
                    for(int i = linearIdx; i < realNumBins; i += numWorkers)
                        cupla::atomicAdd(
                            worker.getAcc(),
                            &(gBins[i]),
                            float_64(shBin[i]),
                            ::alpaka::hierarchy::Blocks{});
                });
        }
    };

    template<class ParticlesType>
    class BinEnergyParticles : public plugins::multi::IInstance
    {
    private:
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
                return std::shared_ptr<IInstance>(new BinEnergyParticles<ParticlesType>(help, id, cellDescription));
            }

            // find all valid filter for the current used species
            template<typename T>
            using Op = typename particles::traits::GenerateSolversIfSpeciesEligible<T, ParticlesType>::type;
            using EligibleFilters = pmacc::mp_flatten<pmacc::mp_transform<Op, particles::filter::AllParticleFilters>>;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod = {"period", "enable plugin [for each n-th step]"};
            plugins::multi::Option<std::string> filter = {"filter", "particle filter: "};
            plugins::multi::Option<int> numBins = {"binCount", "number of bins for the energy range", 1024};
            plugins::multi::Option<float_X> minEnergy_keV = {"minEnergy", "minEnergy[in keV]", 0.0};
            plugins::multi::Option<float_X> maxEnergy_keV = {"maxEnergy", "maxEnergy[in keV]"};

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
                numBins.registerHelp(desc, masterPrefix + prefix);
                minEnergy_keV.registerHelp(desc, masterPrefix + prefix);
                maxEnergy_keV.registerHelp(desc, masterPrefix + prefix);
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
                if(notifyPeriod.size() != maxEnergy_keV.size())
                    throw std::runtime_error(
                        name + ": parameter maxEnergy and period are not used the same number of times");

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

            std::string const name = "BinEnergyParticles";
            //! short description of the plugin
            std::string const description = "calculate a energy histogram of a species";
            //! prefix used for command line arguments
            std::string const prefix = ParticlesType::FrameType::getName() + std::string("_energyHistogram");
        };

        std::unique_ptr<GridBuffer<float_64, DIM1>> gBins;
        MappingDesc* m_cellDescription = nullptr;

        std::string filename;

        std::vector<float_64> binReduced;

        int numBins;
        int realNumBins;
        /* variables for energy limits of the histogram in keV */
        float_X minEnergy_keV;
        float_X maxEnergy_keV;

        std::ofstream outFile;

        /* only rank 0 create a file */
        bool writeToFile = false;

        mpi::MPIReduce reduce;

        std::shared_ptr<Help> m_help;
        size_t m_id;

    public:
        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        BinEnergyParticles(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            filename = m_help->getOptionPrefix() + "_" + m_help->filter.get(m_id) + ".dat";

            numBins = m_help->numBins.get(m_id);

            if(numBins <= 0)
            {
                throw std::runtime_error(
                    std::string("[Plugin] [") + m_help->getOptionPrefix() + "] error since "
                    + m_help->getOptionPrefix() + ".binCount) must be > 0 (input " + std::to_string(numBins)
                    + " bins)");
            }

            minEnergy_keV = m_help->minEnergy_keV.get(m_id);
            maxEnergy_keV = m_help->maxEnergy_keV.get(m_id);

            realNumBins = numBins + 2;

            /* create an array of float_64 on gpu und host */
            gBins = std::make_unique<GridBuffer<float_64, DIM1>>(DataSpace<DIM1>(realNumBins));
            binReduced.resize(realNumBins, 0.0);

            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());
            if(writeToFile)
                openNewFile();

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }

        ~BinEnergyParticles() override
        {
            if(writeToFile)
            {
                outFile.flush();
                outFile << std::endl; /* now all data are written to file */
                if(outFile.fail())
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close();
            }
        }

        void notify(uint32_t currentStep) override
        {
            calBinEnergyParticles<CORE + BORDER>(currentStep);
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
        /* Open a New Output File
         *
         * Must only be called by the rank with writeToFile == true
         */
        void openNewFile()
        {
            outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
            if(!outFile)
            {
                std::cerr << "[Plugin] [" << m_help->getOptionPrefix() << "] Can't open file '" << filename
                          << "', output disabled" << std::endl;
                writeToFile = false;
            }
            else
            {
                /* create header of the file */
                outFile << "#step <" << minEnergy_keV << " ";
                float_X binEnergy = (maxEnergy_keV - minEnergy_keV) / (float_32) numBins;
                for(int i = 1; i < realNumBins - 1; ++i)
                    outFile << minEnergy_keV + ((float_32) i * binEnergy) << " ";

                outFile << ">" << maxEnergy_keV << " count" << std::endl;
            }
        }

        template<uint32_t AREA>
        void calBinEnergyParticles(uint32_t currentStep)
        {
            gBins->getDeviceBuffer().setValue(0);

            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            /* convert energy values from keV to PIConGPU units */
            float_X const minEnergy = minEnergy_keV * UNITCONV_keV_to_Joule / UNIT_ENERGY;
            float_X const maxEnergy = maxEnergy_keV * UNITCONV_keV_to_Joule / UNIT_ENERGY;

            auto const mapper = makeAreaMapper<AREA>(*m_cellDescription);

            auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();

            auto kernel = PMACC_LOCKSTEP_KERNEL(KernelBinEnergyParticles{}, workerCfg)(
                mapper.getGridDim(),
                realNumBins * sizeof(float_X));

            auto bindKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                gBins->getDeviceBuffer().getDataBox(),
                numBins,
                minEnergy,
                maxEnergy,
                mapper,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                bindKernel);

            gBins->deviceToHost();

            reduce(
                pmacc::math::operation::Add(),
                binReduced.data(),
                gBins->getHostBuffer().getBasePointer(),
                realNumBins,
                mpi::reduceMethods::Reduce());


            if(writeToFile)
            {
                using dbl = std::numeric_limits<float_64>;

                outFile.precision(dbl::digits10);

                /* write data to file */
                float_64 count_particles = 0.0;
                outFile << currentStep << " " << std::scientific; /*  for floating points, ignored for ints */

                for(int i = 0; i < realNumBins; ++i)
                {
                    count_particles += float_64(binReduced[i]);
                    outFile << std::scientific << (binReduced[i]) * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                            << " ";
                }
                /* endl: Flush any step to the file.
                 * Thus, we will have data if the program should crash.
                 */
                outFile << std::scientific << count_particles * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                        << std::endl;
            }
        }
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, BinEnergyParticles<T_UnspecifiedSpecies>>
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
