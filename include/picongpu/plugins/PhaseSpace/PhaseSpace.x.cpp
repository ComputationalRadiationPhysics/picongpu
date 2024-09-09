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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#if(ENABLE_OPENPMD == 1)
#    include "picongpu/simulation_defines.hpp"

#    include "picongpu/particles/filter/filter.hpp"
#    include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#    include "picongpu/plugins/PhaseSpace/AxisDescription.hpp"
#    include "picongpu/plugins/PhaseSpace/DumpHBufferOpenPMD.hpp"
#    include "picongpu/plugins/PhaseSpace/Pair.hpp"
#    include "picongpu/plugins/PhaseSpace/PhaseSpaceFunctors.hpp"
#    include "picongpu/plugins/PluginRegistry.hpp"
#    include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#    include "picongpu/plugins/misc/misc.hpp"
#    include "picongpu/plugins/multi/multi.hpp"

#    include <pmacc/communication/manager_common.hpp>
#    include <pmacc/lockstep/lockstep.hpp>
#    include <pmacc/math/Vector.hpp>
#    include <pmacc/mpi/MPIReduce.hpp>
#    include <pmacc/mpi/reduceMethods/Reduce.hpp>
#    include <pmacc/pluginSystem/INotify.hpp>
#    include <pmacc/traits/HasFlag.hpp>
#    include <pmacc/traits/HasIdentifiers.hpp>

#    include <memory>
#    include <string>
#    include <utility>

#    include <mpi.h>


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
        phaseSpace::Pair<float_X, float_X> axis_p_range;
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
            phaseSpace::Pair<float_X, float_X> axis_p_range;
            MappingDesc cellDesc;

            StartBlockFunctor(
                const TParticlesBox& pb,
                pmacc::DataBox<PitchedBox<float_PS, 2>> phaseSpaceDeviceBox,
                const uint32_t p_dir,
                const phaseSpace::Pair<float_X, float_X>& p_range,
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

                auto functorBlock = FunctorBlock<Species, float_PS, num_pbins, r_dir, T_Filter>(
                    particlesBox,
                    phaseSpaceBox,
                    p_element,
                    axis_p_range,
                    filter);
                PMACC_LOCKSTEP_KERNEL(functorBlock).config(mapper.getGridDim(), particlesBox)(mapper);
            }
        };

    public:
        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        PhaseSpace(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            // unit is m_species c (for a single "real" particle)
            float_X pRangeSingle_unit(frame::getMass<typename Species::FrameType>() * SPEED_OF_LIGHT);

            axis_p_range.first = m_help->momentum_range_min.get(id) * pRangeSingle_unit;
            axis_p_range.second = m_help->momentum_range_max.get(id) * pRangeSingle_unit;
            /* String to Enum conversion */
            uint32_t el_space;
            if(m_help->element_space.get(id) == "x")
                el_space = AxisDescription::x;
            else if(m_help->element_space.get(id) == "y")
                el_space = AxisDescription::y;
            else if(m_help->element_space.get(id) == "z")
                el_space = AxisDescription::z;
            else
                throw PluginException("[Plugin] [" + m_help->getOptionPrefix() + "] space must be x, y or z");

            uint32_t el_momentum = AxisDescription::px;
            if(m_help->element_momentum.get(id) == "px")
                el_momentum = AxisDescription::px;
            else if(m_help->element_momentum.get(id) == "py")
                el_momentum = AxisDescription::py;
            else if(m_help->element_momentum.get(id) == "pz")
                el_momentum = AxisDescription::pz;
            else
                throw PluginException("[Plugin] [" + m_help->getOptionPrefix() + "] momentum must be px, py or pz");

            axis_element.momentum = el_momentum;
            axis_element.space = el_space;

            bool activatePlugin = true;

            if constexpr(simDim == DIM2)
                if(el_space == AxisDescription::z)
                {
                    std::cerr << "[Plugin] [" + m_help->getOptionPrefix() + "] Skip requested output for "
                              << m_help->element_space.get(id) << m_help->element_momentum.get(id) << std::endl;
                    activatePlugin = false;
                }

            if(activatePlugin)
            {
                /** create dir */
                Environment<simDim>::get().Filesystem().createDirectoryWithPermissions("phaseSpace");

                const uint32_t r_element = axis_element.space;

                /* CORE + BORDER elements for spatial bins */
                this->r_bins = this->m_cellDescription->getGridLayout().sizeWithoutGuardND()[r_element];

                auto const num_pbinsToAvoidOdrUse = this->num_pbins;
                this->dBuffer
                    = std::make_unique<HostDeviceBuffer<float_PS, 2>>(DataSpace<2>(num_pbinsToAvoidOdrUse, r_bins));

                /* reduce-add phase space from other GPUs in range [p0;p1]x[r;r+dr]
                 * to "lowest" node in range
                 * e.g.: phase space x-py: reduce-add all nodes with same x range in
                 *                         spatial y and z direction to node with
                 *                         lowest y and z position and same x range
                 */
                pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
                pmacc::math::Size_t<simDim> gpuDim = gc.getGpuNodes();
                pmacc::math::Int<simDim> gpuPos = gc.getPosition();

                /* my plane means: the r_element I am calculating should be 1GPU in width */
                pmacc::math::Size_t<simDim> sizeTransversalPlane(gpuDim);
                sizeTransversalPlane[this->axis_element.space] = 1;

                for(int planePos = 0; planePos <= (int) gpuDim[this->axis_element.space]; ++planePos)
                {
                    auto mpiReduce = std::make_unique<mpi::MPIReduce>();
                    bool isInGroup = (gpuPos[this->axis_element.space] == planePos);

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
                    int myRootRank = gc.getGlobalRank() * this->isPlaneReduceRoot - (!this->isPlaneReduceRoot);

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
                        MPI_COMM_WORLD));

                    /* remove all non-roots (-1 values) */
                    std::sort(planeReduceRootRanks.begin(), planeReduceRootRanks.end());
                    std::vector<int> ranks(
                        std::lower_bound(planeReduceRootRanks.begin(), planeReduceRootRanks.end(), 0),
                        planeReduceRootRanks.end());

                    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
                    MPI_CHECK(MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group));
                    MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, new_group, &commFileWriter));
                    MPI_CHECK(MPI_Group_free(&new_group));
                    MPI_CHECK(MPI_Group_free(&world_group));
                }

                // set how often the plugin should be executed while PIConGPU is running
                Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
            }
        }

        virtual ~PhaseSpace()
        {
            if(commFileWriter != MPI_COMM_NULL)
            {
                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&commFileWriter));
            }
        }

        void notify(uint32_t currentStep) override
        {
            /* reset device buffer */
            dBuffer->getDeviceBuffer().setValue(float_PS(0.0));

            /* calculate local phase space */
            if(this->axis_element.space == AxisDescription::x)
                calcPhaseSpace<AxisDescription::x>(currentStep);
            else if(this->axis_element.space == AxisDescription::y)
                calcPhaseSpace<AxisDescription::y>(currentStep);

            if constexpr(simDim == DIM3)
                if(this->axis_element.space == AxisDescription::z)
                    calcPhaseSpace<AxisDescription::z>(currentStep);

            /* transfer to host */
            this->dBuffer->deviceToHost();
            auto bufferExtent = this->dBuffer->getDeviceBuffer().capacityND();

            auto hReducedBuffer = HostBuffer<float_PS, 2>(this->dBuffer->getDeviceBuffer().capacityND());

            eventSystem::getTransactionEvent().waitForFinished();

            /* reduce-add phase space from other GPUs in range [p0;p1]x[r;r+dr]
             * to "lowest" node in range
             * e.g.: phase space x-py: reduce-add all nodes with same x range in
             *                         spatial y and z direction to node with
             *                         lowest y and z position and same x range
             */
            (*planeReduce)(
                pmacc::math::operation::Add(),
                hReducedBuffer.data(),
                this->dBuffer->getHostBuffer().data(),
                bufferExtent.productOfComponents(),
                mpi::reduceMethods::Reduce());

            eventSystem::getTransactionEvent().waitForFinished();

            /** all non-reduce-root processes are done now */
            if(!this->isPlaneReduceRoot)
                return;

            /* write to file */
            const float_64 UNIT_VOLUME = sim.unit.length() * sim.unit.length() * sim.unit.length();
            const float_64 unit = sim.unit.charge() / UNIT_VOLUME;

            /* (momentum) p range: unit is m_species * c
             *   During the kernels we calculate with a typical single/real
             *   momentum range. Now for the dump the meta information of units
             *   on the p-axis should be scaled to represent single/real particles.
             *   @see PhaseSpaceMulti::pluginLoad( )
             */
            float_64 const pRange_unit = sim.unit.mass() * sim.unit.speed();

            DumpHBuffer dumpHBuffer;

            if(this->commFileWriter != MPI_COMM_NULL)
                dumpHBuffer(
                    hReducedBuffer,
                    this->axis_element,
                    this->axis_p_range,
                    pRange_unit,
                    unit,
                    Species::FrameType::getName() + "_" + m_help->filter.get(m_id),
                    m_help->file_name_extension.get(m_id),
                    m_help->json_config.get(m_id),
                    currentStep,
                    this->commFileWriter);
        }

        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
        }

        template<uint32_t r_dir>
        void calcPhaseSpace(const uint32_t currentStep)
        {
            /* register particle species observer */
            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<Species>(Species::FrameType::getName());

            StartBlockFunctor<r_dir> startBlockFunctor(
                particles->getDeviceParticlesBox(),
                dBuffer->getDeviceBuffer().getDataBox(),
                this->axis_element.momentum,
                this->axis_p_range,
                *this->m_cellDescription);

            auto bindFunctor = std::bind(
                startBlockFunctor,
                // particle filter
                std::placeholders::_1);

            auto idProvider = dc.get<IdProvider>("globalId");

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                idProvider->getDeviceGenerator(),
                bindFunctor);
        }
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

PIC_REGISTER_SPECIES_PLUGIN(
    picongpu::plugins::multi::Master<
        picongpu::PhaseSpace<picongpu::particles::shapes::Counter::ChargeAssignment, boost::mpl::_1>>);
#endif
