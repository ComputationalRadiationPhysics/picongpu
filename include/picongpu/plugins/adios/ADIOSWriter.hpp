/* Copyright 2014-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/static_assert.hpp>
#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/adios/ADIOSWriter.def"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/Option.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/traits/IsFieldDomainBound.hpp"

#include <pmacc/particles/frame_types.hpp>
#include <pmacc/particles/IdProvider.def>
#include <pmacc/assert.hpp>

#include "picongpu/fields/CellType.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/MaxwellSolver/YeePML/Field.hpp"
#include <pmacc/particles/operations/CountParticles.hpp>

#include <pmacc/communication/manager_common.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>
#include "picongpu/simulation/control/MovingWindow.hpp"
#include <pmacc/math/Vector.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/traits/Limits.hpp>

#include "picongpu/plugins/output/IIOBackend.hpp"

#include "picongpu/plugins/adios/WriteMeta.hpp"
#include "picongpu/plugins/adios/WriteSpecies.hpp"
#include "picongpu/plugins/adios/ADIOSCountParticles.hpp"
#include "picongpu/plugins/adios/restart/LoadSpecies.hpp"
#include "picongpu/plugins/adios/restart/RestartFieldLoader.hpp"
#include "picongpu/plugins/adios/NDScalars.hpp"
#include "picongpu/plugins/misc/ComponentNames.hpp"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"

#include <adios.h>
#include <adios_read.h>
#include <adios_error.h>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/filesystem.hpp>
#include <boost/type_traits.hpp>

#if !defined(_WIN32)
#    include <unistd.h>
#endif

#include <sstream>
#include <string>
#include <list>
#include <vector>
#include <cstdint>


namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;


        namespace po = boost::program_options;

        template<unsigned DIM>
        int64_t defineAdiosVar(
            int64_t group_id,
            const char* name,
            const char* path,
            enum ADIOS_DATATYPES type,
            pmacc::math::UInt64<DIM> dimensions,
            pmacc::math::UInt64<DIM> globalDimensions,
            pmacc::math::UInt64<DIM> offset,
            bool compression,
            std::string compressionMethod)
        {
            int64_t var_id = 0;

            std::string const revertedDimensions = dimensions.revert().toString(",", "");
            std::string const revertedGlobalDimensions = globalDimensions.revert().toString(",", "");
            std::string const revertedOffset = offset.revert().toString(",", "");
            var_id = adios_define_var(
                group_id,
                name,
                path,
                type,
                revertedDimensions.c_str(),
                revertedGlobalDimensions.c_str(),
                revertedOffset.c_str());

            if(compression)
            {
                /* enable adios transform layer for variable */
                adios_set_transform(var_id, compressionMethod.c_str());
            }

            log<picLog::INPUT_OUTPUT>("ADIOS: Defined varID=%1% for '%2%' at %3% for %4%/%5% elements") % var_id
                % std::string(name) % offset.toString() % dimensions.toString() % globalDimensions.toString();
            return var_id;
        }

        /** Writes simulation data to adios files.
         *
         * Implements the IIOBackend interface.
         */
        class ADIOSWriter : public IIOBackend
        {
        public:
            struct Help : public plugins::multi::IHelp
            {
                /** creates a instance of ISlave
                 *
                 * @tparam T_Slave type of the interface implementation (must inherit from ISlave)
                 * @param help plugin defined help
                 * @param id index of the plugin, range: [0;help->getNumPlugins())
                 */
                std::shared_ptr<ISlave> create(
                    std::shared_ptr<IHelp>& help,
                    size_t const id,
                    MappingDesc* cellDescription)
                {
                    return std::shared_ptr<ISlave>(new ADIOSWriter(help, id, cellDescription));
                }

                plugins::multi::Option<std::string> notifyPeriod = {"period", "enable ADIOS IO [for each n-th step]"};

                plugins::multi::Option<std::string> source = {"source", "data sources: ", "species_all, fields_all"};

                plugins::multi::Option<std::string> fileName = {"file", "ADIOS output filename (prefix)"};

                std::vector<std::string> allowedDataSources = {"species_all", "fields_all"};

                plugins::multi::Option<uint32_t> numAggregators
                    = {"aggregators", "Number of aggregators [0 == number of MPI processes]", 0u};

                plugins::multi::Option<uint32_t> numOSTs = {"ost", "Number of OST", 1u};

                plugins::multi::Option<uint32_t> disableMeta
                    = {"disable-meta",
                       "Disable online gather and write of a global meta file, can be time consuming (use `bpmeta` "
                       "post-mortem)",
                       0u};

                /* select MPI method, #OSTs and #aggregators */
                plugins::multi::Option<std::string> transportParams
                    = {"transport-params",
                       "additional transport parameters, see ADIOS manual chapter 6.1.5, e.g., "
                       "'random_offset=1;stripe_count=4'",
                       ""};

                plugins::multi::Option<std::string> compression
                    = {"compression", "ADIOS compression method, e.g., zlib (see `adios_config -m` for help)", "none"};

                /** defines if the plugin must register itself to the PMacc plugin system
                 *
                 * true = the plugin is registering it self
                 * false = the plugin is not registering itself (plugin is controlled by another class)
                 */
                bool selfRegister = false;

                template<typename T_TupleVector>
                struct CreateSpeciesFilter
                {
                    using type = plugins::misc::SpeciesFilter<
                        typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<0>>::type,
                        typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<1>>::type>;
                };

                using AllParticlesTimesAllFilters = typename AllCombinations<
                    bmpl::vector<FileOutputParticles, particles::filter::AllParticleFilters>>::type;

                using AllSpeciesFilter =
                    typename bmpl::transform<AllParticlesTimesAllFilters, CreateSpeciesFilter<bmpl::_1>>::type;

                using AllEligibleSpeciesSources =
                    typename bmpl::copy_if<AllSpeciesFilter, plugins::misc::speciesFilter::IsEligible<bmpl::_1>>::type;

                using AllFieldSources = FileOutputFields;

                ///! method used by plugin controller to get --help description
                void registerHelp(
                    boost::program_options::options_description& desc,
                    std::string const& masterPrefix = std::string{})
                {
                    meta::ForEach<AllEligibleSpeciesSources, plugins::misc::AppendName<bmpl::_1>>
                        getEligibleDataSourceNames;
                    getEligibleDataSourceNames(allowedDataSources);

                    meta::ForEach<AllFieldSources, plugins::misc::AppendName<bmpl::_1>> appendFieldSourceNames;
                    appendFieldSourceNames(allowedDataSources);

                    // string list with all possible particle sources
                    std::string concatenatedSourceNames = plugins::misc::concatenateToString(allowedDataSources, ", ");

                    notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                    source.registerHelp(desc, masterPrefix + prefix, std::string("[") + concatenatedSourceNames + "]");
                    fileName.registerHelp(desc, masterPrefix + prefix);

                    expandHelp(desc, "");
                    selfRegister = true;
                }

                void expandHelp(
                    boost::program_options::options_description& desc,
                    std::string const& masterPrefix = std::string{})
                {
                    numAggregators.registerHelp(desc, masterPrefix + prefix);
                    numOSTs.registerHelp(desc, masterPrefix + prefix);
                    disableMeta.registerHelp(desc, masterPrefix + prefix);
                    transportParams.registerHelp(desc, masterPrefix + prefix);
                    compression.registerHelp(desc, masterPrefix + prefix);
                }

                void validateOptions()
                {
                    if(selfRegister)
                    {
                        if(notifyPeriod.empty() || fileName.empty())
                            throw std::runtime_error(name + ": parameter period and file must be defined");

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

                size_t getNumPlugins() const
                {
                    if(selfRegister)
                        return notifyPeriod.size();
                    else
                        return 1;
                }

                std::string getDescription() const
                {
                    return description;
                }

                std::string getOptionPrefix() const
                {
                    return prefix;
                }

                std::string getName() const
                {
                    return name;
                }

                std::string const name = "ADIOSWriter";
                //! short description of the plugin
                std::string const description = "dump simulation data with ADIOS";
                //! prefix used for command line arguments
                std::string const prefix = "adios";
            };

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
             * Write calculated fields to adios file.
             */
            template<typename T_Field>
            struct GetFields
            {
            private:
                using ValueType = typename T_Field::ValueType;
                using ComponentType = typename GetComponentsType<ValueType>::type;

            public:
                HDINLINE void operator()(ThreadParams* params)
                {
#ifndef __CUDA_ARCH__
                    DataConnector& dc = Environment<simDim>::get().DataConnector();

                    auto field = dc.get<T_Field>(T_Field::getName());
                    params->gridLayout = field->getGridLayout();
                    const bool isDomainBound = traits::IsFieldDomainBound<T_Field>::value;

                    PICToAdios<ComponentType> adiosType;
                    ADIOSWriter::template writeField<ComponentType>(
                        params,
                        sizeof(ComponentType),
                        adiosType.type,
                        GetNComponents<ValueType>::value,
                        T_Field::getName(),
                        field->getHostDataBox().getPointer(),
                        isDomainBound);

                    dc.releaseData(T_Field::getName());
#endif
                }
            };

            /** Calculate FieldTmp with given solver and particle species
             * and write them to adios.
             *
             * FieldTmp is calculated on device and than dumped to adios.
             */
            template<typename Solver, typename Species>
            struct GetFields<FieldTmpOperation<Solver, Species>>
            {
                /*
                 * This is only a wrapper function to allow disable nvcc warnings.
                 * Warning: calling a __host__ function from __host__ __device__
                 * function.
                 * Use of PMACC_NO_NVCC_HDWARNING is not possible if we call a virtual
                 * method inside of the method were we disable the warnings.
                 * Therefore we create this method and call a new method were we can
                 * call virtual functions.
                 */
                PMACC_NO_NVCC_HDWARNING
                HDINLINE void operator()(ThreadParams* tparam)
                {
                    this->operator_impl(tparam);
                }

            private:
                typedef typename FieldTmp::ValueType ValueType;
                typedef typename GetComponentsType<ValueType>::type ComponentType;

                /** Create a name for the adios identifier.
                 */
                static std::string getName()
                {
                    return FieldTmpOperation<Solver, Species>::getName();
                }

                HINLINE void operator_impl(ThreadParams* params)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();

                    /*## update field ##*/

                    /*load FieldTmp without copy data to host*/
                    PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
                    auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                    /*load particle without copy particle data to host*/
                    auto speciesTmp = dc.get<Species>(Species::FrameType::getName(), true);

                    fieldTmp->getGridBuffer().getDeviceBuffer().setValue(ValueType::create(0.0));
                    /*run algorithm*/
                    fieldTmp->template computeValue<CORE + BORDER, Solver>(*speciesTmp, params->currentStep);

                    EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
                    __setTransactionEvent(fieldTmpEvent);
                    /* copy data to host that we can write same to disk*/
                    fieldTmp->getGridBuffer().deviceToHost();
                    dc.releaseData(Species::FrameType::getName());
                    /*## finish update field ##*/

                    const uint32_t components = GetNComponents<ValueType>::value;
                    PICToAdios<ComponentType> adiosType;

                    params->gridLayout = fieldTmp->getGridLayout();
                    const bool isDomainBound = traits::IsFieldDomainBound<FieldTmp>::value;
                    /*write data to ADIOS file*/
                    ADIOSWriter::template writeField<ComponentType>(
                        params,
                        sizeof(ComponentType),
                        adiosType.type,
                        components,
                        getName(),
                        fieldTmp->getHostDataBox().getPointer(),
                        isDomainBound);

                    dc.releaseData(FieldTmp::getUniqueId(0));
                }
            };

            template<typename T_Field>
            static void defineFieldVar(
                ThreadParams* params,
                uint32_t nComponents,
                ADIOS_DATATYPES adiosType,
                const std::string name,
                std::vector<float_64> unit,
                std::vector<float_64> unitDimension,
                std::vector<std::vector<float_X>> inCellPosition,
                float_X timeOffset)
            {
                PICToAdios<float_64> adiosDoubleType;
                PICToAdios<float_X> adiosFloatXType;

                auto const componentNames = plugins::misc::getComponentNames(nComponents);

                /* parameter checking */
                PMACC_ASSERT(unit.size() == nComponents);
                PMACC_ASSERT(inCellPosition.size() == nComponents);
                for(uint32_t n = 0; n < nComponents; ++n)
                    PMACC_ASSERT(inCellPosition.at(n).size() == simDim);
                PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units

                const std::string recordName(params->adiosBasePath + std::string(ADIOS_PATH_FIELDS) + name);

                auto fieldsSizeDims = params->fieldsSizeDims;
                auto fieldsGlobalSizeDims = params->fieldsGlobalSizeDims;
                auto fieldsOffsetDims = params->fieldsOffsetDims;

                /* Patch for non-domain-bound fields
                 * This is an ugly fix to allow output of reduced 1d PML buffers
                 */
                if(!traits::IsFieldDomainBound<T_Field>::value)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto field = dc.get<T_Field>(T_Field::getName());
                    fieldsSizeDims = precisionCast<uint64_t>(field->getGridLayout().getDataSpaceWithoutGuarding());
                    dc.releaseData(T_Field::getName());

                    /* Scan the PML buffer local size along all local domains
                     * This code is based on the same operation in hdf5::Field::writeField(),
                     * the same comments apply here
                     */
                    log<picLog::INPUT_OUTPUT>("ADIOS:  (begin) collect PML sizes for %1%") % name;
                    auto& gridController = Environment<simDim>::get().GridController();
                    auto const numRanks = uint64_t{gridController.getGlobalSize()};
                    /* Use domain position-based rank, not MPI rank, to be independent
                     * of the MPI rank assignment scheme
                     */
                    auto const rank = uint64_t{gridController.getScalarPosition()};
                    std::vector<uint64_t> localSizes(2u * numRanks, 0u);
                    uint64_t localSizeInfo[2] = {fieldsSizeDims[0], rank};
                    __getTransactionEvent().waitForFinished();
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
                    log<picLog::INPUT_OUTPUT>("ADIOS:  (end) collect PML sizes for %1%") % name;

                    fieldsGlobalSizeDims = pmacc::math::UInt64<simDim>::create(1);
                    fieldsGlobalSizeDims[0] = globalSize;
                    fieldsOffsetDims = pmacc::math::UInt64<simDim>::create(0);
                    fieldsOffsetDims[0] = globalOffsetFile;
                }

                for(uint32_t c = 0; c < nComponents; c++)
                {
                    std::string datasetName = recordName;
                    if(nComponents > 1)
                        datasetName += "/" + componentNames[c];

                    /* define adios var for field, e.g. field_FieldE_y */
                    const char* path = nullptr;
                    int64_t adiosFieldVarId = defineAdiosVar<simDim>(
                        params->adiosGroupHandle,
                        datasetName.c_str(),
                        path,
                        adiosType,
                        fieldsSizeDims,
                        fieldsGlobalSizeDims,
                        fieldsOffsetDims,
                        true,
                        params->adiosCompression);

                    params->adiosFieldVarIds.push_back(adiosFieldVarId);

                    /* already add the unitSI and further attribute so `adios_group_size`
                     * calculates the reservation for the buffer correctly */
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        params->adiosGroupHandle,
                        "position",
                        datasetName.c_str(),
                        adiosFloatXType.type,
                        simDim,
                        &(*inCellPosition.at(c).begin())));

                    ADIOS_CMD(adios_define_attribute_byvalue(
                        params->adiosGroupHandle,
                        "unitSI",
                        datasetName.c_str(),
                        adiosDoubleType.type,
                        1,
                        &unit.at(c)));
                }

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "unitDimension",
                    recordName.c_str(),
                    adiosDoubleType.type,
                    7,
                    &(*unitDimension.begin())));

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "timeOffset",
                    recordName.c_str(),
                    adiosFloatXType.type,
                    1,
                    &timeOffset));

                const std::string geometry("cartesian");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "geometry",
                    recordName.c_str(),
                    adios_string,
                    1,
                    (void*) geometry.c_str()));

                const std::string dataOrder("C");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "dataOrder",
                    recordName.c_str(),
                    adios_string,
                    1,
                    (void*) dataOrder.c_str()));

                if(simDim == DIM2)
                {
                    const char* axisLabels[] = {"y", "x"}; // 2D: F[y][x]
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        params->adiosGroupHandle,
                        "axisLabels",
                        recordName.c_str(),
                        adios_string_array,
                        simDim,
                        axisLabels));
                }
                if(simDim == DIM3)
                {
                    const char* axisLabels[] = {"z", "y", "x"}; // 3D: F[z][y][x]
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        params->adiosGroupHandle,
                        "axisLabels",
                        recordName.c_str(),
                        adios_string_array,
                        simDim,
                        axisLabels));
                }

                // cellSize is {x, y, z} but fields are F[z][y][x]
                std::vector<float_X> gridSpacing(simDim, 0.0);
                for(uint32_t d = 0; d < simDim; ++d)
                    gridSpacing.at(simDim - 1 - d) = cellSize[d];

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "gridSpacing",
                    recordName.c_str(),
                    adiosFloatXType.type,
                    simDim,
                    &(*gridSpacing.begin())));

                /* globalSlideOffset due to gpu slides between origin at time step 0
                 * and origin at current time step
                 * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
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

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "gridGlobalOffset",
                    recordName.c_str(),
                    adiosDoubleType.type,
                    simDim,
                    &(*gridGlobalOffset.begin())));

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "gridUnitSI",
                    recordName.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_LENGTH));

                const std::string fieldSmoothing("none");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "fieldSmoothing",
                    recordName.c_str(),
                    adios_string,
                    1,
                    (void*) fieldSmoothing.c_str()));
            }

            /**
             * Collect field sizes to set adios group size.
             */
            template<typename T>
            struct CollectFieldsSizes
            {
            public:
                typedef typename T::ValueType ValueType;
                typedef typename T::UnitValueType UnitType;
                typedef typename GetComponentsType<ValueType>::type ComponentType;

                static std::vector<float_64> getUnit()
                {
                    UnitType unit = T::getUnit();
                    return createUnit(unit, T::numComponents);
                }

                HDINLINE void operator()(ThreadParams* params)
                {
#ifndef __CUDA_ARCH__
                    const uint32_t components = T::numComponents;

                    auto localSize = params->window.localDimensions.size;
                    /* Patch for non-domain-bound fields
                     * This is an ugly fix to allow output of reduced 1d PML buffers,
                     * that are the same size on each domain.
                     * This code is to be replaced with the openPMD output plugin soon.
                     */
                    if(!traits::IsFieldDomainBound<T>::value)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto field = dc.get<T>(T::getName());
                        localSize = field->getGridLayout().getDataSpaceWithoutGuarding();
                        dc.releaseData(T::getName());
                    }

                    // adios buffer size for this dataset (all components)
                    uint64_t localGroupSize = localSize.productOfComponents() * sizeof(ComponentType) * components;

                    params->adiosGroupSize += localGroupSize;

                    // convert in a std::vector of std::vector format for writeField API
                    const traits::FieldPosition<fields::CellType, T> fieldPos;

                    std::vector<std::vector<float_X>> inCellPosition;
                    for(uint32_t n = 0; n < T::numComponents; ++n)
                    {
                        std::vector<float_X> inCellPositonComponent;
                        for(uint32_t d = 0; d < simDim; ++d)
                            inCellPositonComponent.push_back(fieldPos()[n][d]);
                        inCellPosition.push_back(inCellPositonComponent);
                    }

                    /** \todo check if always correct at this point, depends on solver
                     *        implementation */
                    const float_X timeOffset = 0.0;

                    PICToAdios<ComponentType> adiosType;
                    defineFieldVar<T>(
                        params,
                        components,
                        adiosType.type,
                        T::getName(),
                        getUnit(),
                        T::getUnitDimension(),
                        inCellPosition,
                        timeOffset);
#endif
                }
            };

            /**
             * Collect field sizes to set adios group size.
             * Specialization.
             */
            template<typename Solver, typename Species>
            struct CollectFieldsSizes<FieldTmpOperation<Solver, Species>>
            {
            public:
                PMACC_NO_NVCC_HDWARNING
                HDINLINE void operator()(ThreadParams* tparam)
                {
                    this->operator_impl(tparam);
                }

            private:
                typedef typename FieldTmp::ValueType ValueType;
                typedef typename FieldTmp::UnitValueType UnitType;
                typedef typename GetComponentsType<ValueType>::type ComponentType;

                /** Create a name for the adios identifier.
                 */
                static std::string getName()
                {
                    return FieldTmpOperation<Solver, Species>::getName();
                }

                /** Get the unit for the result from the solver*/
                static std::vector<float_64> getUnit()
                {
                    UnitType unit = FieldTmp::getUnit<Solver>();
                    const uint32_t components = GetNComponents<ValueType>::value;
                    return createUnit(unit, components);
                }

                HINLINE void operator_impl(ThreadParams* params)
                {
                    const uint32_t components = GetNComponents<ValueType>::value;

                    auto localSize = params->window.localDimensions.size;
                    /* Patch for non-domain-bound fields
                     * This is an ugly fix to allow output of reduced 1d PML buffers,
                     * that are the same size on each domain.
                     * This code is to be replaced with the openPMD output plugin soon.
                     */
                    if(!traits::IsFieldDomainBound<FieldTmp>::value)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto field = dc.get<FieldTmp>(FieldTmp::getName());
                        localSize = field->getGridLayout().getDataSpaceWithoutGuarding();
                        dc.releaseData(FieldTmp::getName());
                    }

                    // adios buffer size for this dataset (all components)
                    uint64_t localGroupSize = localSize.productOfComponents() * sizeof(ComponentType) * components;

                    params->adiosGroupSize += localGroupSize;

                    /*wrap in a one-component vector for writeField API*/
                    const traits::FieldPosition<fields::CellType, FieldTmp> fieldPos;

                    std::vector<std::vector<float_X>> inCellPosition;
                    std::vector<float_X> inCellPositonComponent;
                    for(uint32_t d = 0; d < simDim; ++d)
                        inCellPositonComponent.push_back(fieldPos()[0][d]);
                    inCellPosition.push_back(inCellPositonComponent);

                    /** \todo check if always correct at this point, depends on solver
                     *        implementation */
                    const float_X timeOffset = 0.0;

                    PICToAdios<ComponentType> adiosType;
                    defineFieldVar<FieldTmp>(
                        params,
                        components,
                        adiosType.type,
                        getName(),
                        getUnit(),
                        FieldTmp::getUnitDimension<Solver>(),
                        inCellPosition,
                        timeOffset);
                }
            };

        public:
            /** constructor
             *
             * @param help instance of the class Help
             * @param id index of this plugin instance within help
             * @param cellDescription PIConGPu cell description information for kernel index mapping
             */
            ADIOSWriter(std::shared_ptr<plugins::multi::IHelp>& help, size_t const id, MappingDesc* cellDescription)
                : m_help(std::static_pointer_cast<Help>(help))
                , m_id(id)
                , m_cellDescription(cellDescription)
                , outputDirectory("bp")
                , lastSpeciesSyncStep(pmacc::traits::limits::Max<uint32_t>::value)
            {
                mThreadParams.adiosAggregators = m_help->numAggregators.get(id);
                mThreadParams.adiosOST = m_help->numOSTs.get(id);
                mThreadParams.adiosDisableMeta = m_help->disableMeta.get(id);
                mThreadParams.adiosTransportParams = m_help->transportParams.get(id);
                mThreadParams.adiosCompression = m_help->compression.get(id);

                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                /* It is important that we never change the mpi_pos after this point
                 * because we get problems with the restart.
                 * Otherwise we do not know which gpu must load the ghost parts around
                 * the sliding window.
                 */
                mpi_pos = gc.getPosition();
                mpi_size = gc.getGpuNodes();

                /* if number of aggregators is not set we use all mpi process as aggregator*/
                if(mThreadParams.adiosAggregators == 0)
                    mThreadParams.adiosAggregators = mpi_size.productOfComponents();

                if(m_help->selfRegister)
                {
                    std::string notifyPeriod = m_help->notifyPeriod.get(id);
                    /* only register for notify callback when .period is set on command line */
                    if(!notifyPeriod.empty())
                    {
                        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                        /** create notify directory */
                        Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(outputDirectory);
                    }
                }

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                /* Initialize adios library */
                mThreadParams.adiosComm = MPI_COMM_NULL;
                MPI_CHECK(MPI_Comm_dup(gc.getCommunicator().getMPIComm(), &(mThreadParams.adiosComm)));
                mThreadParams.adiosBufferInitialized = false;

                /* select MPI method, #OSTs and #aggregators */
                std::stringstream strMPITransportParams;
                strMPITransportParams << "num_aggregators=" << mThreadParams.adiosAggregators
                                      << ";num_ost=" << mThreadParams.adiosOST;
                /* create meta file offline/post-mortem with bpmeta */
                if(mThreadParams.adiosDisableMeta)
                    strMPITransportParams << ";have_metadata_file=0";
                /* additional, uncovered transport parameters, e.g.,
                 * use system-defaults for striping per aggregated file */
                if(!mThreadParams.adiosTransportParams.empty())
                    strMPITransportParams << ";" << mThreadParams.adiosTransportParams;

                mpiTransportParams = strMPITransportParams.str();
            }

            virtual ~ADIOSWriter()
            {
                if(mThreadParams.adiosComm != MPI_COMM_NULL)
                {
                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    __getTransactionEvent().waitForFinished();
                    MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&(mThreadParams.adiosComm)));
                }
            }

            void notify(uint32_t currentStep)
            {
                // notify is only allowed if the plugin is not controlled by the class Checkpoint
                assert(m_help->selfRegister);

                __getTransactionEvent().waitForFinished();

                std::string filename = m_help->fileName.get(m_id);

                /* if file name is relative, prepend with common directory */
                if(boost::filesystem::path(filename).has_root_path())
                    mThreadParams.adiosFilename = filename;
                else
                    mThreadParams.adiosFilename = outputDirectory + "/" + filename;

                /* window selection */
                mThreadParams.window = MovingWindow::getInstance().getWindow(currentStep);
                mThreadParams.isCheckpoint = false;
                dumpData(currentStep);
            }

            virtual void restart(uint32_t restartStep, std::string const& restartDirectory)
            {
                /* ISlave restart interface is not needed becase IIOBackend
                 * restart interface is used
                 */
            }

            virtual void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory)
            {
                /* ISlave checkpoint interface is not needed becase IIOBackend
                 * checkpoint interface is used
                 */
            }

            void dumpCheckpoint(
                const uint32_t currentStep,
                const std::string& checkpointDirectory,
                const std::string& checkpointFilename)
            {
                // checkpointing is only allowed if the plugin is controlled by the class Checkpoint
                assert(!m_help->selfRegister);

                __getTransactionEvent().waitForFinished();
                /* if file name is relative, prepend with common directory */
                if(boost::filesystem::path(checkpointFilename).has_root_path())
                    mThreadParams.adiosFilename = checkpointFilename;
                else
                    mThreadParams.adiosFilename = checkpointDirectory + "/" + checkpointFilename;

                mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);
                mThreadParams.isCheckpoint = true;

                dumpData(currentStep);
            }

            void doRestart(
                const uint32_t restartStep,
                const std::string& restartDirectory,
                const std::string& constRestartFilename,
                const uint32_t restartChunkSize)
            {
                // restart is only allowed if the plugin is controlled by the class Checkpoint
                assert(!m_help->selfRegister);

                // allow to modify the restart file name
                std::string restartFilename{constRestartFilename};

                std::stringstream adiosPathBase;
                adiosPathBase << ADIOS_PATH_ROOT << restartStep << "/";
                mThreadParams.adiosBasePath = adiosPathBase.str();
                // mThreadParams.isCheckpoint = isCheckpoint;
                mThreadParams.currentStep = restartStep;
                mThreadParams.cellDescription = m_cellDescription;

                /** one could try ADIOS_READ_METHOD_BP_AGGREGATE too which might
                 *  be beneficial for re-distribution on a different number of GPUs
                 *    would need: - `export chunk_size=SIZE # in MB`
                 *                - `mpiTransportParams.c_str()` in `adios_read_init_method`
                 */
                ADIOS_CMD(adios_read_init_method(
                    ADIOS_READ_METHOD_BP,
                    mThreadParams.adiosComm,
                    "verbose=3;abort_on_error;"));

                /* if restartFilename is relative, prepend with restartDirectory */
                if(!boost::filesystem::path(restartFilename).has_root_path())
                {
                    restartFilename = restartDirectory + std::string("/") + restartFilename;
                }

                std::stringstream strFname;
                strFname << restartFilename << "_" << mThreadParams.currentStep << ".bp";

                const std::string filename = strFname.str();

                // adios_read_open( fname, method, comm, lock_mode, timeout_sec )
                log<picLog::INPUT_OUTPUT>("ADIOS: open file: %1%") % filename;

                // when reading in BG_AGGREGATE mode, adios can not distinguish between
                // "file does not exist" and "stream is not (yet) available, so we
                // test it our selves
                if(!boost::filesystem::exists(strFname.str()))
                    throw std::runtime_error("ADIOS: File does not exist.");

                /* <0 sec: wait forever
                 * >=0 sec: return immediately if stream is not available */
                float_32 timeout = 0.0f;
                mThreadParams.fp = adios_read_open(
                    filename.c_str(),
                    ADIOS_READ_METHOD_BP,
                    mThreadParams.adiosComm,
                    ADIOS_LOCKMODE_CURRENT,
                    timeout);

                /* stream reading is tricky, see ADIOS manual section 8.11.1 */
                while(adios_errno == err_file_not_found)
                {
                    /** \todo add c++11 platform independent sleep */
#if !defined(_WIN32)
                    /* give the file system 1s of peace and quiet */
                    usleep(1e6);
#endif
                    mThreadParams.fp = adios_read_open(
                        filename.c_str(),
                        ADIOS_READ_METHOD_BP,
                        mThreadParams.adiosComm,
                        ADIOS_LOCKMODE_CURRENT,
                        timeout);
                }
                if(adios_errno == err_end_of_stream)
                    /* could not read full stream */
                    throw std::runtime_error("ADIOS: Stream terminated too early: " + std::string(adios_errmsg()));
                if(mThreadParams.fp == nullptr)
                    throw std::runtime_error("ADIOS: Error opening stream: " + std::string(adios_errmsg()));

                /* ADIOS types */
                AdiosUInt32Type adiosUInt32Type;

                /* load number of slides to initialize MovingWindow */
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) read attr (%1% available)") % mThreadParams.fp->nattrs;
                void* slidesPtr = nullptr;
                int slideSize;
                enum ADIOS_DATATYPES slidesType;
                const std::string simSlidesPath = mThreadParams.adiosBasePath + std::string("sim_slides");
                ADIOS_CMD(
                    adios_get_attr(mThreadParams.fp, simSlidesPath.c_str(), &slidesType, &slideSize, &slidesPtr));

                uint32_t slides = *((uint32_t*) slidesPtr);
                log<picLog::INPUT_OUTPUT>("ADIOS: value of sim_slides = %1%") % slides;

                PMACC_ASSERT(slidesType == adiosUInt32Type.type);
                PMACC_ASSERT(slideSize == sizeof(uint32_t)); // uint32_t in bytes

                void* lastStepPtr = nullptr;
                int lastStepSize;
                enum ADIOS_DATATYPES lastStepType;
                const std::string iterationPath = mThreadParams.adiosBasePath + std::string("iteration");
                ADIOS_CMD(adios_get_attr(
                    mThreadParams.fp,
                    iterationPath.c_str(),
                    &lastStepType,
                    &lastStepSize,
                    &lastStepPtr));
                uint32_t lastStep = *((uint32_t*) lastStepPtr);
                log<picLog::INPUT_OUTPUT>("ADIOS: value of iteration = %1%") % lastStep;

                PMACC_ASSERT(lastStepType == adiosUInt32Type.type);
                PMACC_ASSERT(lastStep == restartStep);

                /* apply slides to set gpus to last/written configuration */
                log<picLog::INPUT_OUTPUT>("ADIOS: Setting slide count for moving window to %1%") % slides;
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
                meta::ForEach<FileCheckpointFields, LoadFields<bmpl::_1>> forEachLoadFields;
                forEachLoadFields(&mThreadParams);

                /* load all particles */
                meta::ForEach<FileCheckpointParticles, LoadSpecies<bmpl::_1>> forEachLoadSpecies;
                forEachLoadSpecies(&mThreadParams, restartChunkSize);

                IdProvider<simDim>::State idProvState;
                ReadNDScalars<uint64_t, uint64_t>()(
                    mThreadParams,
                    "picongpu/idProvider/startId",
                    &idProvState.startId,
                    "maxNumProc",
                    &idProvState.maxNumProc);
                ReadNDScalars<uint64_t>()(mThreadParams, "picongpu/idProvider/nextId", &idProvState.nextId);
                log<picLog::INPUT_OUTPUT>("Setting next free id on current rank: %1%") % idProvState.nextId;
                IdProvider<simDim>::setState(idProvState);

                /* free memory allocated in ADIOS calls */
                free(slidesPtr);
                free(lastStepPtr);

                // avoid deadlock between not finished pmacc tasks and mpi calls in adios
                __getTransactionEvent().waitForFinished();

                /* clean shut down: close file and finalize */
                adios_release_step(mThreadParams.fp);
                ADIOS_CMD(adios_read_close(mThreadParams.fp));
                ADIOS_CMD(adios_read_finalize_method(ADIOS_READ_METHOD_BP));
            }

        private:
            void endAdios()
            {
                /* Finalize adios library */
                ADIOS_CMD(adios_finalize(Environment<simDim>::get().GridController().getCommunicator().getRank()));

                __deleteArray(mThreadParams.fieldBfr);
            }

            void beginAdios(const std::string adiosFilename)
            {
                std::stringstream full_filename;
                full_filename << adiosFilename << "_" << mThreadParams.currentStep << ".bp";

                mThreadParams.fullFilename = full_filename.str();
                mThreadParams.adiosFileHandle = ADIOS_INVALID_HANDLE;

                // Note: here we always allocate for the domain-bound fields
                mThreadParams.fieldBfr = new float_X[mThreadParams.window.localDimensions.size.productOfComponents()];

                std::stringstream adiosPathBase;
                adiosPathBase << ADIOS_PATH_ROOT << mThreadParams.currentStep << "/";
                mThreadParams.adiosBasePath = adiosPathBase.str();

                ADIOS_CMD(adios_init_noxml(mThreadParams.adiosComm));
            }

            /**
             * Notification for dump or checkpoint received
             *
             * @param currentStep current simulation step
             */
            void dumpData(uint32_t currentStep)
            {
                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                mThreadParams.cellDescription = m_cellDescription;
                mThreadParams.currentStep = currentStep;

                for(uint32_t i = 0; i < simDim; ++i)
                {
                    mThreadParams.localWindowToDomainOffset[i] = 0;
                    if(mThreadParams.window.globalDimensions.offset[i] > localDomain.offset[i])
                    {
                        mThreadParams.localWindowToDomainOffset[i]
                            = mThreadParams.window.globalDimensions.offset[i] - localDomain.offset[i];
                    }
                }

                /* copy species only one time per timestep to the host */
                if(lastSpeciesSyncStep != currentStep)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();

                    /* synchronizes the MallocMCBuffer to the host side */
                    dc.get<MallocMCBuffer<DeviceHeap>>(MallocMCBuffer<DeviceHeap>::getName());

                    /* here we are copying all species to the host side since we
                     * can not say at this point if this time step will need all of them
                     * for sure (checkpoint) or just some user-defined species (dump)
                     */
                    meta::ForEach<FileCheckpointParticles, CopySpeciesToHost<bmpl::_1>> copySpeciesToHost;
                    copySpeciesToHost();
                    lastSpeciesSyncStep = currentStep;
                    dc.releaseData(MallocMCBuffer<DeviceHeap>::getName());
                }

                beginAdios(mThreadParams.adiosFilename);

                writeAdios((void*) &mThreadParams, mpiTransportParams);

                endAdios();
            }

            template<typename ComponentType>
            static void writeField(
                ThreadParams* params,
                const uint32_t sizePtrType,
                ADIOS_DATATYPES adiosType,
                const uint32_t nComponents,
                const std::string name,
                void* ptr,
                const bool isDomainBound)
            {
                log<picLog::INPUT_OUTPUT>("ADIOS: write field: %1% %2% %3%") % name % nComponents % ptr;

                const bool fieldTypeCorrect(boost::is_same<ComponentType, float_X>::value);
                PMACC_CASSERT_MSG(Precision_mismatch_in_Field_Components__ADIOS, fieldTypeCorrect);

                /* data to describe source buffer */
                GridLayout<simDim> field_layout = params->gridLayout;
                DataSpace<simDim> field_full = field_layout.getDataSpace();
                DataSpace<simDim> field_no_guard = params->window.localDimensions.size;
                DataSpace<simDim> field_guard = field_layout.getGuard() + params->localWindowToDomainOffset;
                float_X* dstBuffer = params->fieldBfr;

                /* Patch for non-domain-bound fields
                 * This is an ugly fix to allow output of reduced 1d PML buffers,
                 * that are the same size on each domain.
                 * This code is to be replaced with the openPMD output plugin soon.
                 */
                std::vector<float_X> nonDomainBoundStorage;
                if(!isDomainBound)
                {
                    field_no_guard = field_layout.getDataSpaceWithoutGuarding();
                    field_guard = field_layout.getGuard();
                    /* Since params->fieldBfr allocation was of different size,
                     * for this case allocate a new chunk for memory for dstBuffer
                     */
                    nonDomainBoundStorage.resize(field_no_guard.productOfComponents());
                    dstBuffer = nonDomainBoundStorage.data();
                }

                /* write the actual field data */
                for(uint32_t d = 0; d < nComponents; d++)
                {
                    const size_t plane_full_size = field_full[1] * field_full[0] * nComponents;
                    const size_t plane_no_guard_size = field_no_guard[1] * field_no_guard[0];

                    /* copy strided data from source to temporary buffer
                     *
                     * \todo use d1Access as in `include/plugins/hdf5/writer/Field.hpp`
                     */
                    const int maxZ = simDim == DIM3 ? field_no_guard[2] : 1;
                    const int guardZ = simDim == DIM3 ? field_guard[2] : 0;
                    for(int z = 0; z < maxZ; ++z)
                    {
                        for(int y = 0; y < field_no_guard[1]; ++y)
                        {
                            const size_t base_index_src
                                = (z + guardZ) * plane_full_size + (y + field_guard[1]) * field_full[0] * nComponents;

                            const size_t base_index_dst = z * plane_no_guard_size + y * field_no_guard[0];

                            for(int x = 0; x < field_no_guard[0]; ++x)
                            {
                                size_t index_src = base_index_src + (x + field_guard[0]) * nComponents + d;
                                size_t index_dst = base_index_dst + x;

                                dstBuffer[index_dst] = ((float_X*) ptr)[index_src];
                            }
                        }
                    }

                    /* Write the actual field data. The id is on the front of the list. */
                    if(params->adiosFieldVarIds.empty())
                        throw std::runtime_error("Cannot write field (var id list is empty)");

                    int64_t adiosFieldVarId = *(params->adiosFieldVarIds.begin());
                    params->adiosFieldVarIds.pop_front();
                    ADIOS_CMD(adios_write_byid(params->adiosFileHandle, adiosFieldVarId, dstBuffer));
                }
            }

            template<typename T_ParticleFilter>
            struct CallCountParticles
            {
                void operator()(const std::vector<std::string>& vectorOfDataSourceNames, ThreadParams* params)
                {
                    bool const containsDataSource
                        = plugins::misc::containsObject(vectorOfDataSourceNames, T_ParticleFilter::getName());

                    if(containsDataSource)
                    {
                        ADIOSCountParticles<T_ParticleFilter> count;
                        count(params);
                    }
                }
            };

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
            struct CallCollectFieldsSizes
            {
                void operator()(const std::vector<std::string>& vectorOfDataSourceNames, ThreadParams* params)
                {
                    bool const containsDataSource
                        = plugins::misc::containsObject(vectorOfDataSourceNames, T_Fields::getName());

                    if(containsDataSource)
                    {
                        CollectFieldsSizes<T_Fields> count;
                        count(params);
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

            void* writeAdios(void* p_args, std::string mpiTransportParams)
            {
                // synchronize, because following operations will be blocking anyway
                ThreadParams* threadParams = (ThreadParams*) (p_args);
                threadParams->adiosGroupSize = 0;

                /* y direction can be negative for first gpu */
                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                DataSpace<simDim> particleOffset(localDomain.offset);
                particleOffset.y() -= threadParams->window.globalDimensions.offset.y();

                // do not generate statistics for variables on the fly
                ADIOS_STATISTICS_FLAG noStatistics = adios_stat_no;

                /* create adios group for fields without statistics */
                const std::string iterationPath = threadParams->adiosBasePath + std::string("iteration");
                ADIOS_CMD(adios_declare_group(
                    &(threadParams->adiosGroupHandle),
                    ADIOS_GROUP_NAME,
                    iterationPath.c_str(),
                    noStatistics));

                /* select MPI method, #OSTs and #aggregators */
                ADIOS_CMD(adios_select_method(
                    threadParams->adiosGroupHandle,
                    "MPI_AGGREGATE",
                    mpiTransportParams.c_str(),
                    ""));

                threadParams->fieldsOffsetDims = precisionCast<uint64_t>(localDomain.offset);

                /* write created variable values */
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    /* dimension 1 is y and is the direction of the moving window (if any) */
                    if(1 == d)
                    {
                        uint64_t offset
                            = std::max(0, localDomain.offset.y() - threadParams->window.globalDimensions.offset.y());
                        threadParams->fieldsOffsetDims[d] = offset;
                    }

                    threadParams->fieldsSizeDims[d] = threadParams->window.localDimensions.size[d];
                    threadParams->fieldsGlobalSizeDims[d] = threadParams->window.globalDimensions.size[d];
                }

                std::vector<std::string> vectorOfDataSourceNames;
                if(m_help->selfRegister)
                {
                    std::string dataSourceNames = m_help->source.get(m_id);

                    vectorOfDataSourceNames = plugins::misc::splitString(plugins::misc::removeSpaces(dataSourceNames));
                }

                bool dumpFields = plugins::misc::containsObject(vectorOfDataSourceNames, "fields_all");

                /* collect size information for each field to be written and define
                 * field variables
                 */
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) collecting fields.");
                threadParams->adiosFieldVarIds.clear();
                if(threadParams->isCheckpoint)
                {
                    meta::ForEach<FileCheckpointFields, CollectFieldsSizes<bmpl::_1>> forEachCollectFieldsSizes;
                    forEachCollectFieldsSizes(threadParams);
                }
                else
                {
                    if(dumpFields)
                    {
                        meta::ForEach<FileOutputFields, CollectFieldsSizes<bmpl::_1>> forEachCollectFieldsSizes;
                        forEachCollectFieldsSizes(threadParams);
                    }

                    // move over all field data sources
                    meta::ForEach<typename Help::AllFieldSources, CallCollectFieldsSizes<bmpl::_1>>{}(
                        vectorOfDataSourceNames,
                        threadParams);
                }
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) collecting fields.");

                /* collect size information for all attributes of all species and define
                 * particle variables
                 */
                threadParams->adiosParticleAttrVarIds.clear();
                threadParams->adiosSpeciesIndexVarIds.clear();

                bool dumpAllParticles = plugins::misc::containsObject(vectorOfDataSourceNames, "species_all");

                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) counting particles.");
                if(threadParams->isCheckpoint)
                {
                    meta::ForEach<
                        FileCheckpointParticles,
                        ADIOSCountParticles<plugins::misc::UnfilteredSpecies<bmpl::_1>>>
                        adiosCountParticles;
                    adiosCountParticles(threadParams);
                }
                else
                {
                    // count particles if data source "species_all" is selected
                    if(dumpAllParticles)
                    {
                        // move over all species defined in FileOutputParticles
                        meta::ForEach<
                            FileOutputParticles,
                            ADIOSCountParticles<plugins::misc::UnfilteredSpecies<bmpl::_1>>>
                            adiosCountParticles;
                        adiosCountParticles(threadParams);
                    }

                    // move over all species data sources
                    meta::ForEach<typename Help::AllEligibleSpeciesSources, CallCountParticles<bmpl::_1>>{}(
                        vectorOfDataSourceNames,
                        threadParams);
                }
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) counting particles.");

                auto idProviderState = IdProvider<simDim>::getState();
                WriteNDScalars<uint64_t, uint64_t> writeIdProviderStartId("picongpu/idProvider/startId", "maxNumProc");
                WriteNDScalars<uint64_t, uint64_t> writeIdProviderNextId("picongpu/idProvider/nextId");
                writeIdProviderStartId.prepare(*threadParams, idProviderState.maxNumProc);
                writeIdProviderNextId.prepare(*threadParams);

                // in the past, we had to explicitly estiamte our buffers.
                // this is now done automatically by ADIOS on `adios_write()`
                threadParams->adiosBufferInitialized = true;

                /* open adios file. all variables need to be defined at this point */
                log<picLog::INPUT_OUTPUT>("ADIOS: open file: %1%") % threadParams->fullFilename;
                ADIOS_CMD(adios_open(
                    &(threadParams->adiosFileHandle),
                    ADIOS_GROUP_NAME,
                    threadParams->fullFilename.c_str(),
                    "w",
                    threadParams->adiosComm));

                if(threadParams->adiosFileHandle == ADIOS_INVALID_HANDLE)
                    throw std::runtime_error("ADIOS: Failed to open file.");

                /* attributes written here are pure meta data */
                WriteMeta writeMetaAttributes;
                writeMetaAttributes(threadParams);

                /* set adios group size (total size of all data to be written)
                 * besides the number of bytes for variables, this call also
                 * calculates the overhead of meta data
                 */
                uint64_t adiosTotalSize;
                ADIOS_CMD(
                    adios_group_size(threadParams->adiosFileHandle, threadParams->adiosGroupSize, &adiosTotalSize));

                /* write fields */
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) writing fields.");
                if(threadParams->isCheckpoint)
                {
                    meta::ForEach<FileCheckpointFields, GetFields<bmpl::_1>> forEachGetFields;
                    forEachGetFields(threadParams);
                }
                else
                {
                    if(dumpFields)
                    {
                        meta::ForEach<FileOutputFields, GetFields<bmpl::_1>> forEachGetFields;
                        forEachGetFields(threadParams);
                    }

                    // move over all field data sources
                    meta::ForEach<typename Help::AllFieldSources, CallGetFields<bmpl::_1>>{}(
                        vectorOfDataSourceNames,
                        threadParams);
                }
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) writing fields.");

                /* print all particle species */
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) writing particle species.");
                if(threadParams->isCheckpoint)
                {
                    meta::ForEach<FileCheckpointParticles, WriteSpecies<plugins::misc::SpeciesFilter<bmpl::_1>>>
                        writeSpecies;
                    writeSpecies(threadParams, particleOffset);
                }
                else
                {
                    // dump data if data source "species_all" is selected
                    if(dumpAllParticles)
                    {
                        // move over all species defined in FileOutputParticles
                        meta::ForEach<FileOutputParticles, WriteSpecies<plugins::misc::UnfilteredSpecies<bmpl::_1>>>
                            writeSpecies;
                        writeSpecies(threadParams, particleOffset);
                    }

                    // move over all species data sources
                    meta::ForEach<typename Help::AllEligibleSpeciesSources, CallWriteSpecies<bmpl::_1>>{}(
                        vectorOfDataSourceNames,
                        threadParams,
                        particleOffset);
                }
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) writing particle species.");

                log<picLog::INPUT_OUTPUT>(
                    "ADIOS: Writing IdProvider state (StartId: %1%, NextId: %2%, maxNumProc: %3%)")
                    % idProviderState.startId % idProviderState.nextId % idProviderState.maxNumProc;
                writeIdProviderStartId(*threadParams, idProviderState.startId);
                writeIdProviderNextId(*threadParams, idProviderState.nextId);

                // avoid deadlock between not finished pmacc tasks and mpi calls in adios
                __getTransactionEvent().waitForFinished();

                /* close adios file, most likely the actual write point */
                log<picLog::INPUT_OUTPUT>("ADIOS: closing file: %1%") % threadParams->fullFilename;
                ADIOS_CMD(adios_close(threadParams->adiosFileHandle));

                /*\todo: copied from adios example, we might not need this ? */
                MPI_CHECK(MPI_Barrier(threadParams->adiosComm));

                return nullptr;
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

    } // namespace adios
} // namespace picongpu
