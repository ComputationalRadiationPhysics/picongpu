/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/PhaseSpace/AxisDescription.hpp"
#include "picongpu/plugins/PhaseSpace/PhaseSpaceFunctors.hpp"
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"

#include <pmacc/communication/manager_common.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/pluginSystem/INotify.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <memory>
#include <string>
#include <utility>

#include <mpi.h>


namespace picongpu
{
    using namespace pmacc;
    namespace po = boost::program_options;

    template<class T_AssignmentFunction, class T_Species>
    class PhaseSpace : public plugins::multi::IInstance
    {
    public:
        typedef T_AssignmentFunction AssignmentFunction;
        typedef T_Species Species;

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
                return std::shared_ptr<IInstance>(
                    new PhaseSpace<T_AssignmentFunction, Species>(help, id, cellDescription));
            }

            // find all valid filter for the current used species
            template<typename T>
            using Op = typename particles::traits::GenerateSolversIfSpeciesEligible<T, Species>::type;
            using EligibleFilters = pmacc::mp_flatten<pmacc::mp_transform<Op, particles::filter::AllParticleFilters>>;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod = {"period", "notify period"};
            plugins::multi::Option<std::string> filter = {"filter", "particle filter: "};

            plugins::multi::Option<std::string> element_space = {"space", "spatial component (x, y, z)"};
            plugins::multi::Option<std::string> element_momentum = {"momentum", "momentum component (px, py, pz)"};
            plugins::multi::Option<float_X> momentum_range_min = {"min", "min range momentum [m_species c]"};
            plugins::multi::Option<float_X> momentum_range_max = {"max", "max range momentum [m_species c]"};

            plugins::multi::Option<std::string> file_name_extension
                = {"ext",
                   "openPMD filename extension (this controls the"
                   "backend picked by the openPMD API)",
                   openPMD::getDefaultExtension().c_str()};

            plugins::multi::Option<std::string> json_config
                = {"json", "advanced (backend) configuration for openPMD in JSON format", "{}"};

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

                element_space.registerHelp(desc, masterPrefix + prefix);
                element_momentum.registerHelp(desc, masterPrefix + prefix);
                momentum_range_min.registerHelp(desc, masterPrefix + prefix);
                momentum_range_max.registerHelp(desc, masterPrefix + prefix);
                file_name_extension.registerHelp(desc, masterPrefix + prefix);
                json_config.registerHelp(desc, masterPrefix + prefix);
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
                if(notifyPeriod.size() != element_space.size())
                    throw std::runtime_error(
                        name + ": parameter space and period are not used the same number of times");
                if(notifyPeriod.size() != element_momentum.size())
                    throw std::runtime_error(
                        name + ": parameter momentum and period are not used the same number of times");
                if(notifyPeriod.size() != momentum_range_min.size())
                    throw std::runtime_error(
                        name + ": parameter min and period are not used the same number of times");
                if(notifyPeriod.size() != momentum_range_max.size())
                    throw std::runtime_error(
                        name + ": parameter max and period are not used the same number of times");

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

            std::string const name = "PhaseSpace";
            //! short description of the plugin
            std::string const description = "create phase space of a species";
            //! prefix used for command line arguments
            std::string const prefix = Species::FrameType::getName() + std::string("_phaseSpace");
        };


    private:
        MappingDesc* m_cellDescription = nullptr;

        /** plot to create: e.g. py, x from element_coordinate/momentum */
        AxisDescription axis_element;
        /** range [pMin : pMax] in m_e c */
        std::pair<float_X, float_X> axis_p_range;
        uint32_t r_bins;

        std::shared_ptr<Help> m_help;
        size_t m_id;

        typedef float_32 float_PS;
        /** depending on the super cells edge size and the PS float type
         *  we use not more than 32KB shared memory
         *  Note: checking the longest edge for all phase space configurations
         *        is a conservative work around until #469 is implemented */
        using SuperCellsLongestEdge = typename pmacc::math::CT::max<SuperCellSize>::type;

        /* Note: the previously used 32 KB shared memory size is not correct
         * for CPUs, as discovered in #3329. As a quick patch, slightly reduce
         * it so that the buffer plus a few small shared memory variables
         * together fit 30 KB as set by default on CPUs. So set to 30 000 bytes.
         */
        static constexpr uint32_t maxShared = 30000;
        static constexpr uint32_t num_pbins = maxShared / (sizeof(float_PS) * SuperCellsLongestEdge::value);

        std::unique_ptr<HostDeviceBuffer<float_PS, 2>> dBuffer;

        /** reduce functor to a single host per plane */
        std::unique_ptr<mpi::MPIReduce> planeReduce;
        bool isPlaneReduceRoot = false;
        /** MPI communicator that contains the root ranks of the \p planeReduce
         *
         * This communicator is used to dump data via openPMD.
         */
        MPI_Comm commFileWriter = MPI_COMM_NULL;

        template<uint32_t r_dir>

        struct StartBlockFunctor
        {
            using TParticlesBox = typename Species::ParticlesBoxType;

            TParticlesBox particlesBox;
            pmacc::DataBox<PitchedBox<float_PS, 2>> phaseSpaceBox;
            uint32_t p_element;
            std::pair<float_X, float_X> axis_p_range;
            MappingDesc cellDesc;

            StartBlockFunctor(
                const TParticlesBox& pb,
                pmacc::DataBox<PitchedBox<float_PS, 2>> phaseSpaceDeviceBox,
                const uint32_t p_dir,
                const std::pair<float_X, float_X>& p_range,
                MappingDesc const& cellDescription)
                : particlesBox(pb)
                , phaseSpaceBox(phaseSpaceDeviceBox)
                , p_element(p_dir)
                , axis_p_range(p_range)
                , cellDesc(cellDescription)
            {
            }

            template<typename T_Filter>
            void operator()(T_Filter const& filter) const
            {
                auto const mapper = makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(cellDesc);

                auto workerCfg = pmacc::lockstep::makeWorkerCfg<TParticlesBox::frameSize>();
                auto functorBlock = FunctorBlock<Species, float_PS, num_pbins, r_dir, T_Filter>(
                    particlesBox,
                    phaseSpaceBox,
                    p_element,
                    axis_p_range,
                    filter);
                PMACC_LOCKSTEP_KERNEL(functorBlock, workerCfg)
                (mapper.getGridDim())(mapper);
            }
        };

    public:
        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        PhaseSpace(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription);
        virtual ~PhaseSpace();

        void notify(uint32_t currentStep) override;

        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
        }

        template<uint32_t Direction>
        void calcPhaseSpace(const uint32_t currentStep);
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_AssignmentFunction, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, PhaseSpace<T_AssignmentFunction, T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                using RequiredIdentifiers = MakeSeq_t<weighting, position<>, momentum>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                using SpeciesHasMass = typename pmacc::traits::HasFlag<FrameType, massRatio<>>::type;
                using SpeciesHasCharge = typename pmacc::traits::HasFlag<FrameType, chargeRatio<>>::type;

                using type = pmacc::mp_and<SpeciesHasIdentifiers, SpeciesHasMass, SpeciesHasCharge>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu

#include "PhaseSpace.tpp"
