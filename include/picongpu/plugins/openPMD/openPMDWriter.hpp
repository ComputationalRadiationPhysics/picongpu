/* Copyright 2014-2022 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz, Alexander Grund, Franz Poeschel,
 *                     Pawel Ordyna, Sergei Bastrakov
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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/particleToGrid/CombinedDerive.def"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/openPMDWriteMeta.hpp"
#include "picongpu/plugins/misc/ComponentNames.hpp"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/IHelp.hpp"
#include "picongpu/plugins/multi/Option.hpp"
#include "picongpu/plugins/openPMD/Json.hpp"
#include "picongpu/plugins/openPMD/NDScalars.hpp"
#include "picongpu/plugins/openPMD/WriteSpecies.hpp"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/plugins/openPMD/restart/LoadSpecies.hpp"
#include "picongpu/plugins/openPMD/restart/RestartFieldLoader.hpp"
#include "picongpu/plugins/output/IIOBackend.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"
#include "picongpu/traits/IsFieldDomainBound.hpp"
#include "picongpu/traits/IsFieldOutputOptional.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/assert.hpp>
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/filesystem.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/meta/AllCombinations.hpp>
#include <pmacc/particles/IdProvider.def>
#include <pmacc/particles/frame_types.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/particles/operations/CountParticles.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>
#include <pmacc/pluginSystem/toSlice.hpp>
#include <pmacc/simulationControl/TimeInterval.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/Limits.hpp>

#include <boost/mpl/placeholders.hpp>

#include <openPMD/openPMD.hpp>

#if !defined(_WIN32)
#    include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib> // getenv
#include <exception>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include <pthread.h>


namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;


        namespace po = boost::program_options;

        template<unsigned DIM>
        ::openPMD::RecordComponent& ThreadParams::initDataset(
            ::openPMD::RecordComponent& recordComponent,
            ::openPMD::Datatype datatype,
            pmacc::math::UInt64<DIM> const& globalDimensions,
            std::string const& datasetName)
        {
            std::vector<uint64_t> v = asStandardVector(globalDimensions);
            ::openPMD::Dataset dataset{datatype, std::move(v)};
            dataset.options = jsonMatcher->get(datasetName);
            recordComponent.resetDataset(std::move(dataset));
            return recordComponent;
        }


        template<typename T_Vec, typename T_Ret>
        T_Ret asStandardVector(T_Vec const& v)
        {
            using __T_Vec = typename std::remove_reference<T_Vec>::type;
            constexpr auto dim = __T_Vec::dim;
            T_Ret res(dim);
            for(unsigned i = 0; i < dim; ++i)
            {
                res[dim - i - 1] = v[i];
            }
            return res;
        }

        ::openPMD::Series& ThreadParams::openSeries(::openPMD::Access at)
        {
            if(!openPMDSeries)
            {
                std::string fullName = fileName + fileInfix + "." + fileExtension;
                log<picLog::INPUT_OUTPUT>("openPMD: open file: %1%") % fullName;
                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                eventSystem::getTransactionEvent().waitForFinished();
                openPMDSeries = std::make_unique<::openPMD::Series>(
                    fullName,
                    at,
                    communicator,
                    /*
                     * The writing routines are configured via the JSON set passed
                     * in --openPMD.json / --checkpoint.openPMD.json, or TOML parameter backend_config.
                     * The reading routines (for restarting from a checkpoint)
                     * are configured via --checkpoint.openPMD.jsonRestart.
                     */
                    at == ::openPMD::Access::READ_ONLY ? jsonRestartParams : jsonMatcher->getDefault());
                if(openPMDSeries->backend() == "MPI_ADIOS1")
                {
                    throw std::runtime_error(R"END(
Using ADIOS1 through PIConGPU's openPMD plugin is not supported.
Please use the openPMD plugin with another backend, such as ADIOS2.
In case your openPMD API supports both ADIOS1 and ADIOS2,
make sure that environment variable OPENPMD_BP_BACKEND is not set to ADIOS1.
                )END");
                }
                if(at == ::openPMD::Access::CREATE)
                {
                    openPMDSeries->setMeshesPath(MESHES_PATH);
                    openPMDSeries->setParticlesPath(PARTICLES_PATH);
                }
                log<picLog::INPUT_OUTPUT>("openPMD: successfully opened file: %1%") % fullName;
                return *openPMDSeries;
            }
            else
            {
                throw std::runtime_error("openPMD: Tried opening a Series while old Series was still "
                                         "active");
            }
        }

        void ThreadParams::closeSeries()
        {
            if(openPMDSeries)
            {
                log<picLog::INPUT_OUTPUT>("openPMD: close file: %1%") % fileName;
                openPMDSeries.reset();
                MPI_Barrier(this->communicator);
                log<picLog::INPUT_OUTPUT>("openPMD: successfully closed file: %1%") % fileName;
            }
            else
            {
                throw std::runtime_error("openPMD: Tried closing a Series that was not active");
            }
        }


        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr<plugins::multi::IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override;
            // defined later since we need openPMDWriter constructor

            plugins::multi::Option<std::string> notifyPeriod = {"period", "enable openPMD IO [for each n-th step]"};
            plugins::multi::Option<std::string> range
                = {"range", "define the output range in cells for each dimension e.g. 1:10,:,42:"};

            plugins::multi::Option<std::string> source = {"source", "data sources: ", "species_all, fields_all"};

            plugins::multi::Option<std::string> tomlSources = {"toml", "specify dynamic data sources via TOML"};

            std::vector<std::string> allowedDataSources = {"species_all", "fields_all"};

            plugins::multi::Option<std::string> fileName = {"file", "openPMD file basename"};

            plugins::multi::Option<std::string> fileNameExtension
                = {"ext",
                   "openPMD filename extension. This controls the"
                   "backend picked by the openPMD API. Available extensions: ["
                       + openPMD::printAvailableExtensions() + "]",
                   openPMD::getDefaultExtension().c_str()};

            plugins::multi::Option<std::string> fileNameInfix
                = {"infix",
                   "openPMD filename infix (use to pick file- or group-based "
                   "layout in openPMD)\nSet to NULL to keep empty (e.g. to pick"
                   " group-based iteration layout). Parameter will be ignored"
                   " if a streaming backend is detected in 'ext' parameter and"
                   " an empty string will be assumed instead.",
                   "_%06T"};

            plugins::multi::Option<std::string> jsonConfig
                = {"json", "advanced (backend) configuration for openPMD in JSON format (used when writing)", "{}"};

            plugins::multi::Option<std::string> jsonRestartConfig
                = {"jsonRestart",
                   "advanced (backend) configuration for openPMD in JSON format (used when reading from a checkpoint)",
                   "{}"};

            plugins::multi::Option<std::string> dataPreparationStrategy
                = {"dataPreparationStrategy",
                   "Strategy for preparation of particle data ('doubleBuffer' or "
                   "'mappedMemory'). Aliases 'adios' and 'hdf5' may be used "
                   "respectively.",
                   "doubleBuffer"};

            /** defines if the plugin must register itself to the PMacc plugin
             * system
             *
             * true = the plugin is registering it self
             * false = the plugin is not registering itself (plugin is
             * controlled by another class)
             */
            bool selfRegister = false;

            template<typename T_TupleVector>
            using CreateSpeciesFilter = plugins::misc::SpeciesFilter<
                pmacc::mp_at<T_TupleVector, pmacc::mp_int<0>>,
                pmacc::mp_at<T_TupleVector, pmacc::mp_int<1>>>;

            using AllParticlesTimesAllFilters
                = pmacc::AllCombinations<FileOutputParticles, particles::filter::AllParticleFilters>;

            using AllSpeciesFilter = pmacc::mp_transform<CreateSpeciesFilter, AllParticlesTimesAllFilters>;

        public:
            using AllEligibleSpeciesSources
                = pmacc::mp_copy_if<AllSpeciesFilter, plugins::misc::speciesFilter::IsEligible>;

            using AllFieldSources = FileOutputFields;

            ///! method used by plugin controller to get --help description
            void registerHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                meta::ForEach<AllEligibleSpeciesSources, plugins::misc::AppendName<boost::mpl::_1>>
                    getEligibleDataSourceNames;
                getEligibleDataSourceNames(allowedDataSources);

                meta::ForEach<AllFieldSources, plugins::misc::AppendName<boost::mpl::_1>> appendFieldSourceNames;
                appendFieldSourceNames(allowedDataSources);

                // string list with all possible particle sources
                std::string concatenatedSourceNames = plugins::misc::concatenateToString(allowedDataSources, ", ");

                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                range.registerHelp(desc, masterPrefix + prefix);
                source.registerHelp(desc, masterPrefix + prefix, std::string("[") + concatenatedSourceNames + "]");
                tomlSources.registerHelp(desc, masterPrefix + prefix);
                fileName.registerHelp(desc, masterPrefix + prefix);

                selfRegister = true;
                expandHelp(desc, "");
            }

            void expandHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                fileNameExtension.registerHelp(desc, masterPrefix + prefix);
                fileNameInfix.registerHelp(desc, masterPrefix + prefix);
                jsonConfig.registerHelp(desc, masterPrefix + prefix);
                dataPreparationStrategy.registerHelp(desc, masterPrefix + prefix);
                if(!selfRegister)
                {
                    jsonRestartConfig.registerHelp(desc, masterPrefix + prefix);
                }
            }

            void validateOptions() override
            {
                if(selfRegister)
                {
                    if(tomlSources.empty() && (notifyPeriod.empty() || fileName.empty()))
                        throw std::runtime_error(
                            name + ": If not defining parameter toml, then parameter period and file must be defined");

                    // check if user passed data source names are valid
                    for(auto const& dataSourceNames : source)
                    {
                        auto vectorOfDataSourceNames
                            = plugins::misc::splitString(plugins::misc::removeSpaces(dataSourceNames));

                        for(auto const& f : vectorOfDataSourceNames)
                        {
                            if(!plugins::misc::containsObject(allowedDataSources, f))
                            {
                                throw std::runtime_error(name + ": unknown data source '" + f + "'");
                            }
                        }
                    }
                }
            }

            size_t getNumPlugins() const override
            {
                if(selfRegister)
                {
                    // If using periods in some instances and TOML sources in others, then the other parameter
                    // must be specified as empty.
                    // Not this method's task to check this though.
                    auto res = tomlSources.size() > notifyPeriod.size() ? tomlSources.size() : notifyPeriod.size();
                    if(res == 0)
                    {
                        std::vector<plugins::multi::Option<std::string>> theseMustBeEmpty{
                            source,
                            fileName,
                            fileNameExtension,
                            fileNameInfix,
                            jsonConfig,
                            dataPreparationStrategy};
                        for(auto const& option : theseMustBeEmpty)
                        {
                            if(option.size() > 0)
                            {
                                throw std::runtime_error(
                                    "[openPMD plugin] Parameter '" + option.getName()
                                    + "' was defined, But neither 'period' nor 'toml' was.");
                            }
                        }
                    }
                    return res;
                }
                else
                    return 1;
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

            std::string const name = "openPMDWriter";
            //! short description of the plugin
            std::string const description = "dump simulation data with openPMD";
            //! prefix used for command line arguments
            std::string const prefix = "openPMD";
        };

        void ThreadParams::initFromConfig(
            Help& help,
            size_t id,
            std::string const& dir,
            std::optional<std::string> file)
        {
            std::string strategyString;
            std::string jsonString;

            switch(m_configurationSource)
            {
            case ConfigurationVia::Toml:
                {
                    std::tie(fileName, fileInfix, fileExtension, strategyString, jsonString)
                        = tomlDataSources->openPMDPluginOptions;
                    break;
                }
            case ConfigurationVia::CommandLine:
                {
                    if(!file.has_value())
                    {
                        /*
                         * If file is empty, then the openPMD plugin is running as a normal IO plugin.
                         * In this case, read the file from command line parameters.
                         * If there is a filename, it is running as a checkpoint and the filename
                         * has been supplied from outside.
                         * We must not read it from the command line since it's not there.
                         * Reason: A checkpoint is triggered by writing something like:
                         * > --checkpoint.file asdf --checkpoint.period 100
                         * ... and NOT by something like:
                         * > --checkpoint.openPMD.file asdf --checkpoint.period 100
                         */
                        /* if file name is relative, prepend with common directory */
                        fileName = help.fileName.get(id);
                    }

                    /*
                     * These two however should always be read because they have default values.
                     */
                    fileExtension = help.fileNameExtension.get(id);
                    fileInfix = help.fileNameInfix.get(id);

                    strategyString = help.dataPreparationStrategy.get(id);
                    jsonString = help.jsonConfig.get(id);
                    jsonRestartParams = help.jsonRestartConfig.get(id);
                    break;
                }
            }

            if(file.has_value())
            {
                // If file was specified as function parameter (i.e. when checkpointing), ignore command line
                // parameters for it
                fileName = file.value();
            }

            /*
             * Enforce group-based iteration layout for streaming backends
             */
            if(fileInfix == "NULL" || fileExtension == "sst")
            {
                fileInfix = "";
            }

            fileName = stdfs::path(fileName).has_root_path() ? fileName : dir + "/" + fileName;
            log<picLog::INPUT_OUTPUT>("openPMD: setting file pattern: %1%%2%.%3%") % fileName % fileInfix
                % fileExtension;

            // Avoid repeatedly parsing the JSON config
            if(!jsonMatcher)
            {
                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                jsonMatcher = AbstractJsonMatcher::construct(jsonString, communicator);
            }

            log<picLog::INPUT_OUTPUT>("openPMD: global JSON config: %1%") % jsonMatcher->getDefault();
            if(jsonRestartParams != "{}")
            {
                log<picLog::INPUT_OUTPUT>("openPMD: global JSON restart config: %1%") % jsonRestartParams;
            }

            {
                if(strategyString == "adios" || strategyString == "doubleBuffer")
                {
                    strategy = WriteSpeciesStrategy::ADIOS;
                }
                else if(strategyString == "hdf5" || strategyString == "mappedMemory")
                {
                    strategy = WriteSpeciesStrategy::HDF5;
                }
                else
                {
                    std::cerr << "Passed dataPreparationStrategy for openPMD"
                                 " plugin is invalid."
                              << std::endl;
                }
            }
        }

        /** Writes simulation data to openPMD.
         *
         * Implements the IIOBackend interface.
         */
        class openPMDWriter : public IIOBackend
        {
        public:
            //! must be implemented by the user
            static std::shared_ptr<plugins::multi::IHelp> getHelp()
            {
                return std::shared_ptr<plugins::multi::IHelp>(new Help{});
            }

        private:
            template<typename UnitType>
            static std::vector<float_64> createUnit(UnitType unit, uint32_t numComponents)
            {
                std::vector<float_64> tmp(numComponents);
                for(uint32_t i = 0; i < numComponents; ++i)
                    tmp[i] = unit[i];
                return tmp;
            }

            /**
             * Write calculated fields to openPMD.
             */
            template<typename T_Field>
            struct GetFields
            {
            private:
                using ValueType = typename T_Field::ValueType;
                using ComponentType =
                    typename pmacc::traits::GetComponentsType<ValueType>::type; // FIXME(bgruber): pmacc::traits::
                                                                                // needed because of nvcc 11.0 bug
                using UnitType = typename T_Field::UnitValueType;

            public:
                static std::vector<float_64> getUnit()
                {
                    UnitType unit = T_Field::getUnit();
                    return createUnit(unit, T_Field::numComponents);
                }

                HDINLINE void operator()(ThreadParams* params)
                {
#ifndef __CUDA_ARCH__
                    DataConnector& dc = Environment<simDim>::get().DataConnector();

                    // Skip optional fields
                    if(traits::IsFieldOutputOptional<T_Field>::value && !dc.hasId(T_Field::getName()))
                        return;
                    auto field = dc.get<T_Field>(T_Field::getName());
                    field->synchronize();
                    bool const isDomainBound = traits::IsFieldDomainBound<T_Field>::value;

                    const traits::FieldPosition<fields::CellType, T_Field> fieldPos;

                    std::vector<std::vector<float_X>> inCellPosition;
                    for(uint32_t n = 0; n < T_Field::numComponents; ++n)
                    {
                        std::vector<float_X> inCellPositonComponent;
                        for(uint32_t d = 0; d < simDim; ++d)
                            inCellPositonComponent.push_back(fieldPos()[n][d]);
                        inCellPosition.push_back(inCellPositonComponent);
                    }

                    /** \todo check if always correct at this point, depends on
                     * solver implementation */
                    const float_X timeOffset = 0.0;

                    openPMDWriter::writeField<ComponentType>(
                        params,
                        GetNComponents<ValueType>::value,
                        T_Field::getName(),
                        *field,
                        getUnit(),
                        T_Field::getUnitDimension(),
                        std::move(inCellPosition),
                        timeOffset,
                        isDomainBound);
#endif
                }
            };

            /** Calculate FieldTmp with given solver, particle species, and filter
             * and write them to openPMD.
             *
             * FieldTmp is calculated on device and then dumped to openPMD.
             */
            template<typename Solver, typename Species, typename Filter>
            struct GetFields<FieldTmpOperation<Solver, Species, Filter>>
            {
                /*
                 * This is only a wrapper function to allow disable nvcc warnings.
                 * Warning: calling a __host__ function from __host__ __device__
                 * function.
                 * Use of PMACC_NO_NVCC_HDWARNING is not possible if we call a
                 * virtual method inside of the method were we disable the warnings.
                 * Therefore we create this method and call a new method were we can
                 * call virtual functions.
                 */
                PMACC_NO_NVCC_HDWARNING
                HDINLINE void operator()(ThreadParams* tparam)
                {
                    this->operator_impl(tparam);
                }

            private:
                using UnitType = typename FieldTmp::UnitValueType;
                using ValueType = typename FieldTmp::ValueType;
                /*
                 * Do not change the following lines.
                 * NVCC 11.6 seems to have a parser bug and it does not understand the short form:
                 * `using ComponentType = typename GetComponentsType<ValueType>`
                 * more info: https://github.com/ComputationalRadiationPhysics/picongpu/pull/4006
                 */
                using GetComponentsTypeValueType
                    = pmacc::traits::GetComponentsType<ValueType>; // FIXME(bgruber): pmacc::traits:: needed for
                                                                   // nvcc 11.0
                using ComponentType = typename GetComponentsTypeValueType::type;

                /** Get the unit for the result from the solver*/
                static std::vector<float_64> getUnit()
                {
                    UnitType unit = FieldTmp::getUnit<Solver>();
                    const uint32_t components = GetNComponents<ValueType>::value;
                    return createUnit(unit, components);
                }

                /** Create a name for the openPMD identifier.
                 */
                static std::string getName()
                {
                    return FieldTmpOperation<Solver, Species, Filter>::getName();
                }

                HINLINE void operator_impl(ThreadParams* params)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();

                    /*## update field ##*/

                    /*load FieldTmp without copy data to host*/
                    constexpr uint32_t requiredExtraSlots
                        = particles::particleToGrid::RequiredExtraSlots<Solver>::type::value;
                    PMACC_CASSERT_MSG(
                        _please_allocate_at_least_one_or_two_when_using_combined_attributes_FieldTmp_in_memory_param,
                        fieldTmpNumSlots >= 1u + requiredExtraSlots);
                    auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0));
                    // compute field values
                    auto event
                        = particles::particleToGrid::ComputeFieldValue<CORE + BORDER, Solver, Species, Filter>()(
                            *fieldTmp,
                            params->currentStep,
                            1u);
                    // wait for unfinished asynchronous communication
                    if(event.has_value())
                        eventSystem::setTransactionEvent(*event);
                    /* copy data to host that we can write same to disk*/
                    fieldTmp->getGridBuffer().deviceToHost();
                    /*## finish update field ##*/

                    const uint32_t components = GetNComponents<ValueType>::value;

                    /*wrap in a one-component vector for writeField API*/
                    const traits::FieldPosition<typename fields::CellType, FieldTmp> fieldPos;

                    std::vector<std::vector<float_X>> inCellPosition;
                    std::vector<float_X> inCellPositonComponent;
                    for(uint32_t d = 0; d < simDim; ++d)
                        inCellPositonComponent.push_back(fieldPos()[0][d]);
                    inCellPosition.push_back(inCellPositonComponent);

                    /** \todo check if always correct at this point, depends on
                     * solver implementation */
                    const float_X timeOffset = 0.0;

                    bool const isDomainBound = traits::IsFieldDomainBound<FieldTmp>::value;
                    /*write data to openPMD Series*/
                    openPMDWriter::template writeField<ComponentType>(
                        params,
                        components,
                        getName(),
                        *fieldTmp,
                        getUnit(),
                        FieldTmp::getUnitDimension<Solver>(),
                        std::move(inCellPosition),
                        timeOffset,
                        isDomainBound);
                }
            };

            /** Write random number generator states as a unitless scalar field
             *
             * Note: writeField() cannot be easily used inside this function, since states are custom types.
             * Instead, we reinterpret states as a byte (char) array and store using openPMD API directly.
             *
             * Only suitable for the currently implemented RNG storage: one state per cell, no guards.
             * If this doesn't hold, the implementation can't work as is, and an exception will be thrown.
             */
            HINLINE void writeRngStates(ThreadParams* params)
            {
                DataConnector& dc = Environment<simDim>::get().DataConnector();
                using RNGProvider = pmacc::random::RNGProvider<simDim, random::Generator>;
                auto rngProvider = dc.get<RNGProvider>(RNGProvider::getName());
                // Start copying data to host
                rngProvider->synchronize();
                auto const name = rngProvider->getName();

                ::openPMD::Iteration iteration = params->openPMDSeries->writeIterations()[params->currentStep];
                ::openPMD::Mesh mesh = iteration.meshes[name];

                auto const unitDimension = std::vector<float_64>(7, 0.0);
                auto const timeOffset = 0.0_X;
                writeFieldAttributes(params, unitDimension, timeOffset, mesh);

                ::openPMD::MeshRecordComponent mrc = mesh[::openPMD::RecordComponent::SCALAR];
                std::string datasetName = params->openPMDSeries->meshesPath() + name;

                // rng states are always of the domain size therefore query sizes from domain information
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                pmacc::math::UInt64<simDim> recordLocalSizeDims = subGrid.getLocalDomain().size;
                pmacc::math::UInt64<simDim> recordOffsetDims = subGrid.getLocalDomain().offset;
                pmacc::math::UInt64<simDim> recordGlobalSizeDims = subGrid.getGlobalDomain().size;

                if(recordLocalSizeDims != rngProvider->getSize())
                    throw std::runtime_error("openPMD: RNG state can't be written due to not matching size");

                // Reinterpret state as chars, it must be bitwise-copyable for it
                using ReinterpretedType = char;
                // The fast-moving axis size (x in PIConGPU) had to be adjusted accordingly
                using ValueType = RNGProvider::Buffer::ValueType;
                recordLocalSizeDims[0] *= sizeof(ValueType);
                recordGlobalSizeDims[0] *= sizeof(ValueType);
                recordOffsetDims[0] *= sizeof(ValueType);

                params->initDataset<simDim>(
                    mrc,
                    ::openPMD::determineDatatype<ReinterpretedType>(),
                    recordGlobalSizeDims,
                    datasetName);

                // define record component level attributes
                auto const inCellPosition = std::vector<float_X>(simDim, 0.0_X);
                mrc.setPosition(inCellPosition);
                auto const unit = 1.0_X;
                mrc.setUnitSI(unit);

                auto& buffer = rngProvider->getStateBuffer();
                // getPointer() will wait for device->host transfer
                ValueType* nativePtr = buffer.getHostBuffer().getPointer();
                ReinterpretedType* rawPtr = reinterpret_cast<ReinterpretedType*>(nativePtr);
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                mrc.storeChunkRaw(rawPtr, asStandardVector(recordOffsetDims), asStandardVector(recordLocalSizeDims));
#else
                mrc.storeChunk(
                    ::openPMD::shareRaw(rawPtr),
                    asStandardVector(recordOffsetDims),
                    asStandardVector(recordLocalSizeDims));
#endif
                flushSeries(*params->openPMDSeries, PreferredFlushTarget::Disk);
            }

            /** Implementation of loading random number generator states
             *
             * Note: LoadFields cannot be easily used inside this function, since states are custom types.
             * Instead, we load states as a byte (char) array and reinterpret.
             * This matches how they are written in writeRngStates().
             *
             * Only suitable for the currently implemented RNG storage: one state per cell, no guards.
             * If this doesn't hold, the implementation can't work as is, and an exception will be thrown.
             */
            HINLINE void loadRngStatesImpl(ThreadParams* params)
            {
                DataConnector& dc = Environment<simDim>::get().DataConnector();
                using RNGProvider = pmacc::random::RNGProvider<simDim, random::Generator>;
                auto rngProvider = dc.get<RNGProvider>(RNGProvider::getName());
                auto const name = rngProvider->getName();

                ::openPMD::Iteration iteration = params->openPMDSeries->writeIterations()[params->currentStep];
                ::openPMD::Mesh mesh = iteration.meshes[name];
                ::openPMD::MeshRecordComponent mrc = mesh[::openPMD::RecordComponent::SCALAR];

                // rng states are always of the domain size therefore query sizes from pmacc
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                using VecUInt64 = pmacc::math::UInt64<simDim>;
                VecUInt64 recordLocalSizeDims = subGrid.getLocalDomain().size;
                VecUInt64 recordOffsetDims = subGrid.getLocalDomain().offset;

                if(recordLocalSizeDims != rngProvider->getSize())
                    throw std::runtime_error("openPMD: RNG state can't be loaded due to not matching size");

                // Reinterpret state as chars, it must be bitwise-copyable for it
                using ReinterpretedType = char;
                // The fast-moving axis size (x in PIConGPU) had to be adjusted accordingly
                using ValueType = RNGProvider::Buffer::ValueType;
                recordLocalSizeDims[0] *= sizeof(ValueType);
                recordOffsetDims[0] *= sizeof(ValueType);

                auto& buffer = rngProvider->getStateBuffer();
                ValueType* nativePtr = buffer.getHostBuffer().getPointer();
                ReinterpretedType* rawPtr = reinterpret_cast<ReinterpretedType*>(nativePtr);
                /* Explicit template parameters to asStandardVector required
                 * as we need to change the element type as well
                 */
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                mrc.loadChunkRaw(
                    rawPtr,
                    asStandardVector<VecUInt64, ::openPMD::Offset>(recordOffsetDims),
                    asStandardVector<VecUInt64, ::openPMD::Extent>(recordLocalSizeDims));
#else
                mrc.loadChunk(
                    ::openPMD::shareRaw(rawPtr),
                    asStandardVector<VecUInt64, ::openPMD::Offset>(recordOffsetDims),
                    asStandardVector<VecUInt64, ::openPMD::Extent>(recordLocalSizeDims));
#endif
                params->openPMDSeries->flush();
                // Copy data to device
                rngProvider->syncToDevice();
            }

            /** Load random number generator states
             *
             * In case it triggers an exception in the process, swallow it and do nothing.
             * Then the states will be re-initialized.
             */
            HINLINE void loadRngStates(ThreadParams* params)
            {
                /* Do not enforce it to support older checkpoints.
                 * In case RNG states can't be loaded, they will be default-initialized.
                 * This guard may be removed in the future.
                 */
                try
                {
                    loadRngStatesImpl(&mThreadParams);
                }
                catch(...)
                {
                    log<picLog::INPUT_OUTPUT>(
                        "openPMD: loading RNG states failed, they will be re-initialized instead");
                }
            }

        public:
            /** constructor
             *
             * @param help instance of the class Help
             * @param id index of this plugin instance within help
             * @param cellDescription PIConGPu cell description information for
             * kernel index mapping
             */
            openPMDWriter(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
                : m_help(std::static_pointer_cast<Help>(help))
                , m_id(id)
                , m_cellDescription(cellDescription)
                , outputDirectory("openPMD")
                , lastSpeciesSyncStep(pmacc::traits::limits::Max<uint32_t>::value)
            {
                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                /* It is important that we never change the mpi_pos after this point
                 * because we get problems with the restart.
                 * Otherwise we do not know which gpu must load the ghost parts
                 * around the sliding window.
                 */
                mpi_pos = gc.getPosition();
                mpi_size = gc.getGpuNodes();

                // avoid deadlock between not finished pmacc tasks and mpi blocking
                // collectives
                eventSystem::getTransactionEvent().waitForFinished();
                mThreadParams.communicator = MPI_COMM_NULL;
                MPI_CHECK(MPI_Comm_dup(gc.getCommunicator().getMPIComm(), &(mThreadParams.communicator)));

                if(m_help->selfRegister)
                {
                    /* only register for notify callback when .period is set on
                     * command line */
                    bool tomlSourcesSpecified
                        = m_help->tomlSources.optionDefined(m_id) && not m_help->tomlSources.get(m_id).empty();
                    bool notifyPeriodSpecified
                        = m_help->notifyPeriod.optionDefined(m_id) && not m_help->notifyPeriod.get(m_id).empty();
                    if(tomlSourcesSpecified && not notifyPeriodSpecified)
                    {
                        // Verify that all other parameters are empty for this instance of the plugin
                        std::vector<plugins::multi::Option<std::string>> theseMustBeEmpty{
                            m_help->source,
                            m_help->fileName,
                            m_help->fileNameExtension,
                            m_help->fileNameInfix,
                            m_help->jsonConfig,
                            m_help->dataPreparationStrategy};
                        for(auto const& option : theseMustBeEmpty)
                        {
                            if(option.optionDefined(m_id) && not option.get(m_id).empty())
                            {
                                throw std::runtime_error(
                                    "[openPMD plugin] If using parameter toml, no other parameter may be used (do not "
                                    "define '"
                                    + option.getName() + "').");
                            }
                        }

                        std::string const& tomlSources = m_help->tomlSources.get(id);
                        mThreadParams.m_configurationSource = ConfigurationVia::Toml;

                        mThreadParams.tomlDataSources = std::make_unique<toml::DataSources>(
                            m_help->tomlSources.get(id),
                            m_help->allowedDataSources,
                            mThreadParams.communicator);

                        Environment<>::get().PluginConnector().setNotificationPeriod(
                            this,
                            mThreadParams.tomlDataSources->periods());

                        /** create notify directory */
                        Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(outputDirectory);
                    }
                    else if(not tomlSourcesSpecified && notifyPeriodSpecified)
                    {
                        if(m_help->fileName.empty())
                            throw std::runtime_error("[openPMD plugin] If defining parameter period, then parameter "
                                                     "file must also be defined");

                        std::string const& notifyPeriod = m_help->notifyPeriod.get(id);
                        mThreadParams.m_configurationSource = ConfigurationVia::CommandLine;
                        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                        /** create notify directory */
                        Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(outputDirectory);
                    }
                    else
                    {
                        throw std::runtime_error("[openPMD plugin] Either the notify period or the TOML sources must "
                                                 "be specified, but not both.");
                    }
                }
            }

            virtual ~openPMDWriter()
            {
                if(mThreadParams.communicator != MPI_COMM_NULL)
                {
                    // avoid deadlock between not finished pmacc tasks and mpi
                    // blocking collectives
                    eventSystem::getTransactionEvent().waitForFinished();
                    MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&(mThreadParams.communicator)));
                }
            }


            void notify(uint32_t currentStep) override
            {
                // notify is only allowed if the plugin is not controlled by the
                // class Checkpoint
                assert(m_help->selfRegister);

                eventSystem::getTransactionEvent().waitForFinished();

                mThreadParams.initFromConfig(*m_help, m_id, outputDirectory);

                /* window selection */
                auto simulationOutputWindow = MovingWindow::getInstance().getWindow(currentStep);

                // set default if the user is not providing the parameter.
                std::string selectedRange = ":,:,:";
                if(m_help->range.optionDefined(m_id) && !m_help->range.get(m_id).empty())
                    selectedRange = m_help->range.get(m_id);

                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                mThreadParams.window
                    = plugins::misc::intersectRangeWithWindow(subGrid, simulationOutputWindow, selectedRange);

                mThreadParams.isCheckpoint = false;
                dumpData(currentStep);
            }

            void restart(uint32_t restartStep, std::string const& restartDirectory) override
            {
                /* IInstance restart interface is not needed becase IIOBackend
                 * restart interface is used
                 */
            }

            void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
            {
                /* IInstance checkpoint interface is not needed becase IIOBackend
                 * checkpoint interface is used
                 */
            }

            void dumpCheckpoint(
                const uint32_t currentStep,
                const std::string& checkpointDirectory,
                const std::string& checkpointFilename) override
            {
                // checkpointing is only allowed if the plugin is controlled by the
                // class Checkpoint
                assert(!m_help->selfRegister);

                eventSystem::getTransactionEvent().waitForFinished();
                /* if file name is relative, prepend with common directory */

                mThreadParams.isCheckpoint = true;
                mThreadParams.initFromConfig(*m_help, m_id, checkpointDirectory, checkpointFilename);

                mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);

                dumpData(currentStep);
            }

            /** Checks if a loaded openPMD series is supported for restarting.
             *
             * The openPMD series attribute 'picongpuIOVersionMajor', 'picongpuIOVersionMinor', and the current
             * PIConGPU IO version is compared to find incompatible versions. In case the loaded series is incompatible
             * an runtime error is thrown.
             *
             * @attention Loading a series with older IO major file version as the current PIConGPU IO version is only
             * possible if it is explicitly allowed by this function e.g. in cases the IO plugin is guaranteeing the
             * compatibility. Loading a file with a newer IO major version is not possible.
             *
             * @param series OpenPMD series which is used to load restart data.
             */
            void checkIOFileVersionRestartCompatibility(std::unique_ptr<::openPMD::Series>& series) const
            {
                /* Major version 0 and not having the attribute picongpuIOVersionMajor is handled equally later on.
                 * Major version 0 will never have any other minor version than 1.
                 */
                int ioVersionUsedFileMajor = 0;
                int ioVersionUsedFileMinor = 1;

                if(series->containsAttribute("picongpuIOVersionMajor"))
                {
                    ioVersionUsedFileMajor = series->getAttribute("picongpuIOVersionMajor").get<int>();
                }
                else
                {
                    /* No version available. An old file before the version feature was introduced is loaded.
                     * Use the default version zero in that case.
                     */
                    log<picLog::INPUT_OUTPUT>(
                        "openPMD: Restart from file without attribute 'picongpuIOVersionMajor'.");
                }

                if(series->containsAttribute("picongpuIOVersionMinor"))
                {
                    ioVersionUsedFileMinor = series->getAttribute("picongpuIOVersionMinor").get<int>();
                }
                else if(ioVersionUsedFileMajor != 0)
                {
                    // We require a minor version number only if major version is not 0.
                    throw std::runtime_error(
                        "openPMD: Restart file openPMD series attribute 'picongpuIOVersionMinor' is missing.");
                }

                std::string seriesIoVersion
                    = std::to_string(ioVersionUsedFileMajor) + "." + std::to_string(ioVersionUsedFileMinor);

                std::string picongpuIoVersion
                    = std::to_string(picongpuIOVersionMajor) + "." + std::to_string(picongpuIOVersionMinor);

                log<picLog::INPUT_OUTPUT>("openPMD: Restart file IO version is %1% and PIConGPU's version is %2%")
                    % seriesIoVersion % picongpuIoVersion;

                // handle compatibility between picongpu io file versions
                if(picongpuIOVersionMajor < ioVersionUsedFileMajor)
                {
                    throw std::runtime_error(
                        std::string("openPMD: Restart file IO version ") + seriesIoVersion
                        + " is newer and incompatible to the current supported file format version "
                        + picongpuIoVersion + ".");
                }
                else if(picongpuIOVersionMajor > ioVersionUsedFileMajor)
                {
                    throw std::runtime_error(
                        std::string("openPMD: Restart file IO version ") + seriesIoVersion
                        + " is older and incompatible to the current supported file format version "
                        + picongpuIoVersion + ".");
                }
            }

            void doRestart(
                const uint32_t restartStep,
                const std::string& restartDirectory,
                const std::string& constRestartFilename,
                const uint32_t restartChunkSize) override
            {
                // restart is only allowed if the plugin is controlled by the class
                // Checkpoint
                assert(!m_help->selfRegister);

                mThreadParams.initFromConfig(*m_help, m_id, restartDirectory, constRestartFilename);

                // mThreadParams.isCheckpoint = isCheckpoint;
                mThreadParams.currentStep = restartStep;
                mThreadParams.cellDescription = m_cellDescription;

                mThreadParams.openSeries(::openPMD::Access::READ_ONLY);

                checkIOFileVersionRestartCompatibility(mThreadParams.openPMDSeries);

                ::openPMD::Iteration iteration = mThreadParams.openPMDSeries->iterations[mThreadParams.currentStep];

                /* load number of slides to initialize MovingWindow */
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) read attr (%1% available)") % iteration.numAttributes();


                uint32_t slides = iteration.getAttribute("sim_slides").get<uint32_t>();
                log<picLog::INPUT_OUTPUT>("openPMD: value of sim_slides = %1%") % slides;

                uint32_t lastStep = iteration.getAttribute("iteration").get<uint32_t>();
                log<picLog::INPUT_OUTPUT>("openPMD: value of iteration = %1%") % lastStep;

                PMACC_ASSERT(lastStep == restartStep);

                /* apply slides to set gpus to last/written configuration */
                log<picLog::INPUT_OUTPUT>("openPMD: Setting slide count for moving window to %1%") % slides;
                MovingWindow::getInstance().setSlideCounter(slides, restartStep);

                /* re-distribute the local offsets in y-direction
                 * this will work for restarts with moving window still enabled
                 * and restarts that disable the moving window
                 * \warning enabling the moving window from a checkpoint that
                 *          had no moving window will not work
                 */
                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                gc.setStateAfterSlides(slides);

                /* set window for restart, complete global domain */
                mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(restartStep);
                mThreadParams.localWindowToDomainOffset = DataSpace<simDim>::create(0);

                /* load all fields */
                meta::ForEach<FileCheckpointFields, LoadFields<boost::mpl::_1>> ForEachLoadFields;
                ForEachLoadFields(&mThreadParams);

                /* load all particles */
                meta::ForEach<FileCheckpointParticles, LoadSpecies<boost::mpl::_1>> ForEachLoadSpecies;
                ForEachLoadSpecies(&mThreadParams, restartChunkSize);

                loadRngStates(&mThreadParams);

                IdProvider<simDim>::State idProvState;
                ReadNDScalars<uint64_t, uint64_t>()(
                    mThreadParams,
                    "picongpu",
                    "idProvider",
                    "startId",
                    &idProvState.startId,
                    "maxNumProc",
                    &idProvState.maxNumProc);
                ReadNDScalars<uint64_t>()(mThreadParams, "picongpu", "idProvider", "nextId", &idProvState.nextId);
                log<picLog::INPUT_OUTPUT>("Setting next free id on current rank: %1%") % idProvState.nextId;
                IdProvider<simDim>::setState(idProvState);

                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                eventSystem::getTransactionEvent().waitForFinished();

                // Finalize the openPMD Series by calling its destructor
                mThreadParams.closeSeries();
            }

        private:
            std::vector<std::string> currentDataSources(uint32_t currentStep)
            {
                switch(mThreadParams.m_configurationSource)
                {
                case ConfigurationVia::Toml:
                    return mThreadParams.tomlDataSources->currentDataSources(currentStep);
                case ConfigurationVia::CommandLine:
                    {
                        std::string dataSourceNames = m_help->source.get(m_id);
                        return plugins::misc::splitString(plugins::misc::removeSpaces(dataSourceNames));
                    }
                }
                throw std::runtime_error("Unreachable!");
            }

            void endWrite()
            {
                mThreadParams.fieldBuffer.resize(0);
            }

            void initWrite()
            {
                // fieldBuffer will only be resized if needed
                // in some openPMD backends, it's more efficient to let the backend handle buffer creation
                // (span-based RecordComponent::storeChunk() API)
            }

            /**
             * Notification for dump or checkpoint received
             *
             * @param currentStep current simulation step
             */
            void dumpData(uint32_t currentStep)
            {
                // local offset + extent
                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                mThreadParams.cellDescription = m_cellDescription;
                mThreadParams.currentStep = currentStep;

                for(uint32_t i = 0; i < simDim; ++i)
                {
                    mThreadParams.localWindowToDomainOffset[i]
                        = std::max(0, mThreadParams.window.globalDimensions.offset[i] - localDomain.offset[i]);
                }

#if(PMACC_CUDA_ENABLED == 1 || ALPAKA_ACC_GPU_HIP_ENABLED == 1)
                /* copy species only one time per timestep to the host */
                if(mThreadParams.strategy == WriteSpeciesStrategy::ADIOS && lastSpeciesSyncStep != currentStep)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();

                    /* synchronizes the MallocMCBuffer to the host side */
                    auto mallocMCBuffer = dc.get<MallocMCBuffer<DeviceHeap>>(MallocMCBuffer<DeviceHeap>::getName());
                    mallocMCBuffer->synchronize();


                    /* here we are copying all species to the host side since we
                     * can not say at this point if this time step will need all of
                     * them for sure (checkpoint) or just some user-defined species
                     * (dump)
                     */
                    meta::ForEach<FileCheckpointParticles, CopySpeciesToHost<boost::mpl::_1>> copySpeciesToHost;
                    copySpeciesToHost();
                    lastSpeciesSyncStep = currentStep;
                }
#endif

                TimeIntervall timer;
                timer.toggleStart();
                initWrite();

                write(&mThreadParams, mpiTransportParams);

                endWrite();
                timer.toggleEnd();
                double interval = timer.getInterval();
                mThreadParams.times.push_back(interval);
                double average = std::accumulate(mThreadParams.times.begin(), mThreadParams.times.end(), 0);
                average /= mThreadParams.times.size();
                log<picLog::INPUT_OUTPUT>("openPMD: IO plugin ran for %1% (average: %2%)") % timer.printeTime(interval)
                    % timer.printeTime(average);
            }

            static void writeFieldAttributes(
                ThreadParams* params,
                std::vector<float_64> const& unitDimension,
                float_X timeOffset,
                ::openPMD::Mesh& mesh)
            {
                static constexpr ::openPMD::UnitDimension openPMDUnitDimensions[7]
                    = {::openPMD::UnitDimension::L,
                       ::openPMD::UnitDimension::M,
                       ::openPMD::UnitDimension::T,
                       ::openPMD::UnitDimension::I,
                       ::openPMD::UnitDimension::theta,
                       ::openPMD::UnitDimension::N,
                       ::openPMD::UnitDimension::J};
                std::map<::openPMD::UnitDimension, double> unitMap;
                for(unsigned i = 0; i < 7; ++i)
                {
                    unitMap[openPMDUnitDimensions[i]] = unitDimension[i];
                }

                mesh.setUnitDimension(unitMap);
                mesh.setTimeOffset<float_X>(timeOffset);
                mesh.setGeometry(::openPMD::Mesh::Geometry::cartesian);
                mesh.setDataOrder(::openPMD::Mesh::DataOrder::C);

                if constexpr(simDim == DIM2)
                {
                    std::vector<std::string> axisLabels = {"y", "x"}; // 2D: F[y][x]
                    mesh.setAxisLabels(axisLabels);
                }
                if constexpr(simDim == DIM3)
                {
                    std::vector<std::string> axisLabels = {"z", "y", "x"}; // 3D: F[z][y][x]
                    mesh.setAxisLabels(axisLabels);
                }

                // cellSize is {x, y, z} but fields are F[z][y][x]
                std::vector<float_X> gridSpacing(simDim, 0.0);
                for(uint32_t d = 0; d < simDim; ++d)
                    gridSpacing.at(simDim - 1 - d) = cellSize[d];

                mesh.setGridSpacing(gridSpacing);

                /* globalSlideOffset due to gpu slides between origin at time step 0
                 * and origin at current time step
                 * ATTENTION: offset is globalSlideOffset + picongpu offsets
                 */
                DataSpace<simDim> globalSlideOffset;
                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(params->currentStep);
                globalSlideOffset.y() += numSlides * localDomain.size.y();

                // globalDimensions is {x, y, z} but fields are F[z][y][x]
                std::vector<float_64> gridGlobalOffset(simDim, 0.0);
                for(uint32_t d = 0; d < simDim; ++d)
                    gridGlobalOffset.at(simDim - 1 - d) = float_64(cellSize[d])
                        * float_64(params->window.globalDimensions.offset[d] + globalSlideOffset[d]);

                mesh.setGridGlobalOffset(std::move(gridGlobalOffset));
                mesh.setGridUnitSI(UNIT_LENGTH);
                mesh.setAttribute("fieldSmoothing", "none");
            }

            template<typename ComponentType, typename FieldBuffer>
            static void writeField(
                ThreadParams* params,
                const uint32_t nComponents,
                const std::string name,
                FieldBuffer& buffer,
                std::vector<float_64> unit,
                std::vector<float_64> unitDimension,
                std::vector<std::vector<float_X>> inCellPosition,
                float_X timeOffset,
                bool isDomainBound)
            {
                auto const name_lookup_tpl = plugins::misc::getComponentNames(nComponents);
                ::openPMD::Datatype const openPMDType = ::openPMD::determineDatatype<ComponentType>();

                if(openPMDType == ::openPMD::Datatype::UNDEFINED)
                {
                    throw std::runtime_error(
                        "[openPMD plugin] Trying to write a field of a datatype unknown to openPMD.");
                }

                /* parameter checking */
                PMACC_ASSERT(unit.size() == nComponents);
                PMACC_ASSERT(inCellPosition.size() == nComponents);
                for(uint32_t n = 0; n < nComponents; ++n)
                    PMACC_ASSERT(inCellPosition.at(n).size() == simDim);
                PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units

                log<picLog::INPUT_OUTPUT>("openPMD: write field: %1% %2% %3%") % name % nComponents
                    % buffer.getHostDataBox().getPointer();

                ::openPMD::Iteration iteration = params->openPMDSeries->writeIterations()[params->currentStep];
                ::openPMD::Mesh mesh = iteration.meshes[name];

                // set mesh attributes
                writeFieldAttributes(params, unitDimension, timeOffset, mesh);

                /* data to describe source buffer */
                GridLayout<simDim> bufferGridLayout = buffer.getGridLayout();
                DataSpace<simDim> bufferSize = bufferGridLayout.getDataSpace();

                DataSpace<simDim> localWindowSize = params->window.localDimensions.size;
                DataSpace<simDim> bufferOffset = bufferGridLayout.getGuard() + params->localWindowToDomainOffset;
                std::vector<char>& fieldBuffer = params->fieldBuffer;

                pmacc::math::UInt64<simDim> recordLocalSizeDims = localWindowSize;
                pmacc::math::UInt64<simDim> recordOffsetDims = params->window.localDimensions.offset;
                pmacc::math::UInt64<simDim> recordGlobalSizeDims = params->window.globalDimensions.size;

                /* Patch for non-domain-bound fields
                 * Allow for the output of reduced 1d PML buffer
                 */
                if(!isDomainBound)
                {
                    localWindowSize = bufferGridLayout.getDataSpaceWithoutGuarding();
                    bufferOffset = bufferGridLayout.getGuard();

                    recordLocalSizeDims = precisionCast<uint64_t>(localWindowSize);

                    /* Scan the PML buffer local size along all local domains
                     * This code is based on the same operation in hdf5::Field::writeField(),
                     * the same comments apply here
                     */
                    log<picLog::INPUT_OUTPUT>("openPMD:  (begin) collect PML sizes for %1%") % name;
                    auto& gridController = Environment<simDim>::get().GridController();
                    auto const numRanks = uint64_t{gridController.getGlobalSize()};
                    /* Use domain position-based rank, not MPI rank, to be independent
                     * of the MPI rank assignment scheme
                     */
                    auto const rank = uint64_t{gridController.getScalarPosition()};
                    std::vector<uint64_t> localSizes(2u * numRanks, 0u);
                    uint64_t localSizeInfo[2] = {recordLocalSizeDims[0], rank};
                    eventSystem::getTransactionEvent().waitForFinished();
                    MPI_CHECK(MPI_Allgather(
                        localSizeInfo,
                        2,
                        MPI_UINT64_T,
                        &(*localSizes.begin()),
                        2,
                        MPI_UINT64_T,
                        gridController.getCommunicator().getMPIComm()));
                    uint64_t globalOffsetFile = 0;
                    uint64_t globalSize = 0;
                    for(uint64_t r = 0; r < numRanks; ++r)
                    {
                        globalSize += localSizes.at(2u * r);
                        if(localSizes.at(2u * r + 1u) < rank)
                            globalOffsetFile += localSizes.at(2u * r);
                    }
                    log<picLog::INPUT_OUTPUT>("openPMD:  (end) collect PML sizes for %1%") % name;

                    recordGlobalSizeDims = pmacc::math::UInt64<simDim>::create(1);
                    recordGlobalSizeDims[0] = globalSize;
                    recordOffsetDims = pmacc::math::UInt64<simDim>::create(0);
                    recordOffsetDims[0] = globalOffsetFile;
                }

                auto const numDataPoints = localWindowSize.productOfComponents();

                /* write the actual field data */
                for(uint32_t d = 0; d < nComponents; d++)
                {
                    ::openPMD::MeshRecordComponent mrc
                        = mesh[nComponents > 1 ? name_lookup_tpl[d] : ::openPMD::RecordComponent::SCALAR];
                    std::string datasetName = nComponents > 1
                        ? params->openPMDSeries->meshesPath() + name + "/" + name_lookup_tpl[d]
                        : params->openPMDSeries->meshesPath() + name;

                    params->initDataset<simDim>(mrc, openPMDType, recordGlobalSizeDims, datasetName);

                    // define record component level attributes
                    mrc.setPosition(inCellPosition.at(d));
                    mrc.setUnitSI(unit.at(d));

                    if(numDataPoints == 0)
                    {
                        flushSeries(*params->openPMDSeries, PreferredFlushTarget::Disk);
                        continue;
                    }

                    // ask openPMD to create a buffer for us
                    // in some backends (ADIOS2), this allows avoiding memcopies
                    auto span = storeChunkSpan<ComponentType>(
                        mrc,
                        asStandardVector(recordOffsetDims),
                        asStandardVector(recordLocalSizeDims),
                        [&fieldBuffer](size_t size)
                        {
                            // if there is no special backend support for creating buffers,
                            // reuse the fieldBuffer
                            fieldBuffer.resize(sizeof(ComponentType) * size);
                            return std::shared_ptr<ComponentType>{
                                reinterpret_cast<ComponentType*>(fieldBuffer.data()),
                                [](auto*) {}};
                        });
                    auto dstBuffer = span.currentBuffer();

                    const size_t bufferSizeXYPlane = bufferSize[1] * bufferSize[0] * nComponents;
                    const size_t dateSizeXYPlane = localWindowSize[1] * localWindowSize[0];

                    /* copy strided data from source to temporary buffer
                     *
                     * \todo use d1Access as in
                     * `include/plugins/hdf5/writer/Field.hpp`
                     */
                    const int maxZ = simDim == DIM3 ? localWindowSize[2] : 1;
                    const int guardZ = simDim == DIM3 ? bufferOffset[2] : 0;
                    void* ptr = buffer.getHostDataBox().getPointer();
                    for(int z = 0; z < maxZ; ++z)
                    {
                        for(int y = 0; y < localWindowSize[1]; ++y)
                        {
                            const size_t base_index_src = (z + guardZ) * bufferSizeXYPlane
                                + (y + bufferOffset[1]) * bufferSize[0] * nComponents;

                            const size_t base_index_dst = z * dateSizeXYPlane + y * localWindowSize[0];

                            for(int x = 0; x < localWindowSize[0]; ++x)
                            {
                                size_t index_src = base_index_src + (x + bufferOffset[0]) * nComponents + d;
                                size_t index_dst = base_index_dst + x;

                                dstBuffer[index_dst] = reinterpret_cast<ComponentType*>(ptr)[index_src];
                            }
                        }
                    }

                    flushSeries(*params->openPMDSeries, PreferredFlushTarget::Disk);
                }
            }


            template<typename T_ParticleFilter>
            struct CallWriteSpecies
            {
                template<typename Space>
                void operator()(
                    const std::vector<std::string>& vectorOfDataSourceNames,
                    ThreadParams* params,
                    const Space domainOffset)
                {
                    bool const containsDataSource
                        = plugins::misc::containsObject(vectorOfDataSourceNames, T_ParticleFilter::getName());

                    if(containsDataSource)
                    {
                        WriteSpecies<T_ParticleFilter> writeSpecies;
                        writeSpecies(params, domainOffset);
                    }
                }
            };

            template<typename T_Fields>
            struct CallGetFields
            {
                void operator()(const std::vector<std::string>& vectorOfDataSourceNames, ThreadParams* params)
                {
                    bool const containsDataSource
                        = plugins::misc::containsObject(vectorOfDataSourceNames, T_Fields::getName());

                    if(containsDataSource)
                    {
                        GetFields<T_Fields> getFields;
                        getFields(params);
                    }
                }
            };

            void write(ThreadParams* threadParams, std::string mpiTransportParams)
            {
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                const pmacc::Selection<simDim> localDomain = subGrid.getLocalDomain();
                const pmacc::Selection<simDim> globalDomain = subGrid.getGlobalDomain();
                /* Offset to transform local particle offsets into total offsets for all particles within the
                 * current local domain.
                 * @attention A window can be the full simulation domain or the moving window.
                 */
                DataSpace<simDim> particleToTotalDomainOffset(localDomain.offset + globalDomain.offset);

                std::vector<std::string> vectorOfDataSourceNames;
                if(m_help->selfRegister)
                {
                    vectorOfDataSourceNames = currentDataSources(threadParams->currentStep);
                }

                bool dumpFields = plugins::misc::containsObject(vectorOfDataSourceNames, "fields_all");

                if(threadParams->openPMDSeries)
                {
                    log<picLog::INPUT_OUTPUT>("openPMD: Series still open, reusing");
                    // TODO check for same configuration
                }
                else
                {
                    log<picLog::INPUT_OUTPUT>("openPMD: opening Series %1%") % threadParams->fileName;
                    threadParams->openSeries(::openPMD::Access::CREATE);
                }

                /* attributes written here are pure meta data */
                WriteMeta writeMetaAttributes;
                writeMetaAttributes(
                    *threadParams->openPMDSeries,
                    (*threadParams->openPMDSeries).writeIterations()[threadParams->currentStep],
                    threadParams->currentStep);

                bool dumpAllParticles = plugins::misc::containsObject(vectorOfDataSourceNames, "species_all");

                /* write fields */
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) writing fields.");
                if(threadParams->isCheckpoint)
                {
                    meta::ForEach<FileCheckpointFields, GetFields<boost::mpl::_1>> ForEachGetFields;
                    ForEachGetFields(threadParams);
                }
                else
                {
                    if(dumpFields)
                    {
                        meta::ForEach<FileOutputFields, GetFields<boost::mpl::_1>> ForEachGetFields;
                        ForEachGetFields(threadParams);
                    }

                    // move over all field data sources
                    meta::ForEach<typename Help::AllFieldSources, CallGetFields<boost::mpl::_1>>{}(
                        vectorOfDataSourceNames,
                        threadParams);
                }
                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing fields.");


                /* print all particle species */
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) writing particle species.");
                if(threadParams->isCheckpoint)
                {
                    meta::ForEach<
                        FileCheckpointParticles,
                        WriteSpecies<
                            plugins::misc::SpeciesFilter<boost::mpl::_1>,
                            plugins::misc::UnfilteredSpecies<boost::mpl::_1>>>
                        writeSpecies;
                    writeSpecies(threadParams, particleToTotalDomainOffset);
                }
                else
                {
                    // dump data if data source "species_all" is selected
                    if(dumpAllParticles)
                    {
                        // move over all species defined in FileOutputParticles
                        meta::ForEach<
                            FileOutputParticles,
                            WriteSpecies<plugins::misc::UnfilteredSpecies<boost::mpl::_1>>>
                            writeSpecies;
                        writeSpecies(threadParams, particleToTotalDomainOffset);
                    }

                    // move over all species data sources
                    meta::ForEach<typename Help::AllEligibleSpeciesSources, CallWriteSpecies<boost::mpl::_1>>{}(
                        vectorOfDataSourceNames,
                        threadParams,
                        particleToTotalDomainOffset);
                }
                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing particle species.");

                // No need for random generator states in normal output, only in checkpoints
                if(threadParams->isCheckpoint)
                {
                    log<picLog::INPUT_OUTPUT>("openPMD: ( begin ) writing RNG states.");
                    writeRngStates(threadParams);
                    log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing RNG states.");
                }

                auto idProviderState = IdProvider<simDim>::getState();
                log<picLog::INPUT_OUTPUT>("openPMD: Writing IdProvider state (StartId: %1%, NextId: %2%, "
                                          "maxNumProc: %3%)")
                    % idProviderState.startId % idProviderState.nextId % idProviderState.maxNumProc;

                WriteNDScalars<uint64_t, uint64_t> writeIdProviderStartId(
                    "picongpu",
                    "idProvider",
                    "startId",
                    "maxNumProc");
                WriteNDScalars<uint64_t, uint64_t> writeIdProviderNextId("picongpu", "idProvider", "nextId");
                writeIdProviderStartId(*threadParams, idProviderState.startId, idProviderState.maxNumProc);
                writeIdProviderNextId(*threadParams, idProviderState.nextId);

                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                eventSystem::getTransactionEvent().waitForFinished();
                mThreadParams.openPMDSeries->writeIterations()[mThreadParams.currentStep].close();

                return;
            }

            ThreadParams mThreadParams;

            std::shared_ptr<Help> m_help;
            size_t m_id;

            MappingDesc* m_cellDescription;

            std::string outputDirectory;

            /* select MPI method, #OSTs and #aggregators */
            std::string mpiTransportParams;

            uint32_t lastSpeciesSyncStep;

            DataSpace<simDim> mpi_pos;
            DataSpace<simDim> mpi_size;
        };

        std::shared_ptr<plugins::multi::IInstance> Help::create(
            std::shared_ptr<plugins::multi::IHelp>& help,
            size_t const id,
            MappingDesc* cellDescription)
        {
            return std::shared_ptr<plugins::multi::IInstance>(new openPMDWriter(help, id, cellDescription));
        }

    } // namespace openPMD

    /*
     * Logically, these functions should be defined inside toml.cpp.
     * However, their implementation relies on includes that PIConGPU's
     * structure currently prevents from being included into hostonly files.
     * So, let's NVCC compile their definitions.
     */
    namespace toml
    {
        void writeLog(char const* message, size_t argsc, char const* const* argsv)
        {
            auto logg = log<picLog::INPUT_OUTPUT>(message);
            for(size_t i = 0; i < argsc; ++i)
            {
                logg = logg % argsv[i];
            }
        }

        std::vector<TimeSlice> parseTimeSlice(std::string const& asString)
        {
            std::vector<TimeSlice> res;
            auto parsed = pmacc::pluginSystem::toTimeSlice(asString);
            res.reserve(parsed.size());
            std::transform(
                parsed.begin(),
                parsed.end(),
                std::back_inserter(res),
                [](pmacc::pluginSystem::Slice timeSlice) -> TimeSlice {
                    return {timeSlice.values[0], timeSlice.values[1], timeSlice.values[2]};
                });
            return res;
        }
    } // namespace toml
} // namespace picongpu
