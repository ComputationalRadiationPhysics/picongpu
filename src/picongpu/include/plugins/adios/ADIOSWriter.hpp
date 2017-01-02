/**
 * Copyright 2014-2016 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pthread.h>
#include <sstream>
#include <string>
#include <list>
#include <vector>

#include "pmacc_types.hpp"
#include "simulation_types.hpp"
#include "plugins/adios/ADIOSWriter.def"

#include "particles/frame_types.hpp"
#include "particles/IdProvider.def"
#include "assert.hpp"

#include <adios.h>
#include <adios_read.h>
#include <adios_error.h>

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"
#include "particles/operations/CountParticles.hpp"

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/GridLayout.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "math/Vector.hpp"
#include "particles/memory/buffers/MallocMCBuffer.hpp"
#include "traits/Limits.hpp"

#include "plugins/ILightweightPlugin.hpp"
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
#include <unistd.h>
#endif

#include "plugins/adios/WriteMeta.hpp"
#include "plugins/adios/WriteSpecies.hpp"
#include "plugins/adios/ADIOSCountParticles.hpp"
#include "plugins/adios/restart/LoadSpecies.hpp"
#include "plugins/adios/restart/RestartFieldLoader.hpp"
#include "plugins/adios/NDScalars.hpp"
#include "plugins/common/stringHelpers.hpp"


namespace picongpu
{

namespace adios
{

using namespace PMacc;



namespace po = boost::program_options;

template <unsigned DIM>
int64_t defineAdiosVar(int64_t group_id,
                       const char * name,
                       const char * path,
                       enum ADIOS_DATATYPES type,
                       PMacc::math::UInt64<DIM> dimensions,
                       PMacc::math::UInt64<DIM> globalDimensions,
                       PMacc::math::UInt64<DIM> offset,
                       bool compression,
                       std::string compressionMethod)
{
    /* disable compression if this rank writes no data */
    bool canCompress = true;
    for (size_t i = 0; i < DIM; ++i)
    {
        if (dimensions[i] == 0 || globalDimensions[i] == 0)
        {
            canCompress = false;
        }
    }

    int64_t var_id = 0;
    if ((DIM == 1) && (globalDimensions.productOfComponents() == 1)) {
        /* scalars need empty size strings */
        var_id = adios_define_var(
            group_id, name, path, type, 0, 0, 0);
    } else {
        var_id = adios_define_var(
            group_id, name, path, type,
            dimensions.revert().toString(",", "").c_str(),
            globalDimensions.revert().toString(",", "").c_str(),
            offset.revert().toString(",", "").c_str());
    }

    if (compression && canCompress)
    {
        /* enable zlib compression for variable, default compression level */
        adios_set_transform(var_id, compressionMethod.c_str());
    }

    log<picLog::INPUT_OUTPUT > ("ADIOS: Defined varID=%1% for '%2%' at %3% for %4%/%5% elements") %
                var_id % std::string(name) % offset.toString() % dimensions.toString() % globalDimensions.toString();
    return var_id;
}

/**
 * Writes simulation data to adios files.
 * Implements the ILightweightPlugin interface.
 */
class ADIOSWriter : public ILightweightPlugin
{
private:

    template<typename UnitType>
    static std::vector<float_64> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<float_64> tmp(numComponents);
        for (uint32_t i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }

    /**
     * Write calculated fields to adios file.
     */
    template< typename T >
    struct GetFields
    {
    private:
        typedef typename T::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

    public:

        HDINLINE void operator()(ThreadParams* params)
        {
#ifndef __CUDA_ARCH__
            DataConnector &dc = Environment<simDim>::get().DataConnector();

            T* field = &(dc.getData<T > (T::getName()));
            params->gridLayout = field->getGridLayout();

            PICToAdios<ComponentType> adiosType;
            ADIOSWriter::template writeField<ComponentType>(params,
                       sizeof(ComponentType),
                       adiosType.type,
                       GetNComponents<ValueType>::value,
                       T::getName(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getName());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     * and write them to adios.
     *
     * FieldTmp is calculated on device and than dumped to adios.
     */
    template< typename Solver, typename Species >
    struct GetFields<FieldTmpOperation<Solver, Species> >
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
            std::stringstream str;
            str << Species::FrameType::getName();
            str << "_";
            str << Solver().getName();
            return str.str();
        }

        HINLINE void operator_impl(ThreadParams* params)
        {
            DataConnector &dc = Environment<>::get().DataConnector();

            /*## update field ##*/

            /*load FieldTmp without copy data to host*/
            FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FieldTmp::getName(), true));
            /*load particle without copy particle data to host*/
            Species* speciesTmp = &(dc.getData<Species >(Species::FrameType::getName(), true));

            fieldTmp->getGridBuffer().getDeviceBuffer().setValue(ValueType::create(0.0));
            /*run algorithm*/
            fieldTmp->computeValue < CORE + BORDER, Solver > (*speciesTmp, params->currentStep);

            EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(fieldTmpEvent);
            /* copy data to host that we can write same to disk*/
            fieldTmp->getGridBuffer().deviceToHost();
            dc.releaseData(Species::FrameType::getName());
            /*## finish update field ##*/

            const uint32_t components = GetNComponents<ValueType>::value;
            PICToAdios<ComponentType> adiosType;

            params->gridLayout = fieldTmp->getGridLayout();
            /*write data to ADIOS file*/
            ADIOSWriter::template writeField<ComponentType>(params,
                       sizeof(ComponentType),
                       adiosType.type,
                       components,
                       getName(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FieldTmp::getName());

        }

    };

    static void defineFieldVar(ThreadParams* params,
        uint32_t nComponents, ADIOS_DATATYPES adiosType, const std::string name,
        std::vector<float_64> unit, std::vector<float_64> unitDimension,
        std::vector<std::vector<float_X> > inCellPosition, float_X timeOffset)
    {
        PICToAdios<float_64> adiosDoubleType;
        PICToAdios<float_X> adiosFloatXType;

        const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};

        /* parameter checking */
        PMACC_ASSERT( unit.size() == nComponents );
        PMACC_ASSERT( inCellPosition.size() == nComponents );
        for( uint32_t n = 0; n < nComponents; ++n )
            PMACC_ASSERT( inCellPosition.at(n).size() == simDim );
        PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units

        const std::string recordName( params->adiosBasePath +
            std::string(ADIOS_PATH_FIELDS) + name );

        for( uint32_t c = 0; c < nComponents; c++ )
        {
            std::stringstream datasetName;
            datasetName << recordName;
            if (nComponents > 1)
                datasetName << "/" << name_lookup_tpl[c];

            /* define adios var for field, e.g. field_FieldE_y */
            const char* path = NULL;
            int64_t adiosFieldVarId = defineAdiosVar<simDim>(
                    params->adiosGroupHandle,
                    datasetName.str().c_str(),
                    path,
                    adiosType,
                    params->fieldsSizeDims,
                    params->fieldsGlobalSizeDims,
                    params->fieldsOffsetDims,
                    true,
                    params->adiosCompression);

            params->adiosFieldVarIds.push_back(adiosFieldVarId);

            /* already add the unitSI and further attribute so `adios_group_size`
             * calculates the reservation for the buffer correctly */
            ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
                      "position", datasetName.str().c_str(),
                      adiosFloatXType.type, simDim, &(*inCellPosition.at(c).begin()) ));

            ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
                      "unitSI", datasetName.str().c_str(),
                      adiosDoubleType.type, 1, &unit.at(c) ));
        }

        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "unitDimension", recordName.c_str(),
            adiosDoubleType.type, 7, &(*unitDimension.begin()) ));

        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "timeOffset", recordName.c_str(),
            adiosFloatXType.type, 1, &timeOffset ));

        const std::string geometry( "cartesian" );
        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "geometry", recordName.c_str(),
            adios_string, 1, (void*)geometry.c_str() ));

        const std::string dataOrder( "C" );
        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "dataOrder", recordName.c_str(),
            adios_string, 1, (void*)dataOrder.c_str() ));

        if( simDim == DIM2 )
        {
            const char* axisLabels[] = {"y", "x"};      // 2D: F[y][x]
            ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
                "axisLabels", recordName.c_str(),
                adios_string_array, simDim, axisLabels ));
        }
        if( simDim == DIM3 )
        {
            const char* axisLabels[] = {"z", "y", "x"}; // 3D: F[z][y][x]
            ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
                "axisLabels", recordName.c_str(),
                adios_string_array, simDim, axisLabels ));
        }

        std::vector<float_X> gridSpacing(simDim, 0.0);
        for( uint32_t d = 0; d < simDim; ++d )
            gridSpacing.at(d) = cellSize[d];

        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "gridSpacing", recordName.c_str(),
            adiosFloatXType.type, simDim, &(*gridSpacing.begin()) ));

        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(params->currentStep);
        globalSlideOffset.y() += numSlides * localDomain.size.y();

        std::vector<float_64> gridGlobalOffset(simDim, 0.0);
        for( uint32_t d = 0; d < simDim; ++d )
            gridGlobalOffset.at(d) =
                float_64(cellSize[d]) *
                float_64(params->window.globalDimensions.offset[d] +
                         globalSlideOffset[d]);

        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "gridGlobalOffset", recordName.c_str(),
            adiosDoubleType.type, simDim, &(*gridGlobalOffset.begin()) ));

        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "gridUnitSI", recordName.c_str(),
            adiosDoubleType.type, 1, (void*)&UNIT_LENGTH ));

        const std::string fieldSmoothing( "none" );
        ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
            "fieldSmoothing", recordName.c_str(),
            adios_string, 1, (void*)fieldSmoothing.c_str() ));
    }

    /**
     * Collect field sizes to set adios group size.
     */
    template< typename T >
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

            // adios buffer size for this dataset (all components)
            uint64_t localGroupSize =
                    params->window.localDimensions.size.productOfComponents() *
                    sizeof(ComponentType) *
                    components;

            params->adiosGroupSize += localGroupSize;

            // convert in a std::vector of std::vector format for writeField API
            const fieldSolver::numericalCellType::traits::FieldPosition<T> fieldPos;

            std::vector<std::vector<float_X> > inCellPosition;
            for( uint32_t n = 0; n < T::numComponents; ++n )
            {
                std::vector<float_X> inCellPositonComponent;
                for( uint32_t d = 0; d < simDim; ++d )
                    inCellPositonComponent.push_back( fieldPos()[n][d] );
                inCellPosition.push_back( inCellPositonComponent );
            }

            /** \todo check if always correct at this point, depends on solver
             *        implementation */
            const float_X timeOffset = 0.0;


            PICToAdios<ComponentType> adiosType;
            defineFieldVar(params, components, adiosType.type, T::getName(), getUnit(),
                T::getUnitDimension(), inCellPosition, timeOffset);
#endif
        }
    };

    /**
     * Collect field sizes to set adios group size.
     * Specialization.
     */
    template< typename Solver, typename Species >
    struct CollectFieldsSizes<FieldTmpOperation<Solver, Species> >
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
            std::stringstream str;
            str << Solver().getName();
            str << "_";
            str << Species::FrameType::getName();
            return str.str();
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

            // adios buffer size for this dataset (all components)
            uint64_t localGroupSize =
                    params->window.localDimensions.size.productOfComponents() *
                    sizeof(ComponentType) *
                    components;

            params->adiosGroupSize += localGroupSize;

            /*wrap in a one-component vector for writeField API*/
            const fieldSolver::numericalCellType::traits::FieldPosition<FieldTmp>
                fieldPos;

            std::vector<std::vector<float_X> > inCellPosition;
            std::vector<float_X> inCellPositonComponent;
            for( uint32_t d = 0; d < simDim; ++d )
                inCellPositonComponent.push_back( fieldPos()[0][d] );
            inCellPosition.push_back( inCellPositonComponent );

            /** \todo check if always correct at this point, depends on solver
             *        implementation */
            const float_X timeOffset = 0.0;

            PICToAdios<ComponentType> adiosType;
            defineFieldVar(params, components, adiosType.type, getName(), getUnit(),
                FieldTmp::getUnitDimension<Solver>(), inCellPosition, timeOffset);
        }

    };

public:

    ADIOSWriter() :
    filename("simData"),
    outputDirectory("bp"),
    checkpointFilename("checkpoint"),
    restartFilename(""), /* set to checkpointFilename by default */
    /* select MPI method, #OSTs and #aggregators */
    mpiTransportParams(""),
    notifyPeriod(0),
    lastSpeciesSyncStep(PMacc::traits::limits::Max<uint32_t>::value)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~ADIOSWriter()
    {

    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("adios.period", po::value<uint32_t > (&notifyPeriod)->default_value(0),
             "enable ADIOS IO [for each n-th step]")
            ("adios.aggregators", po::value<uint32_t >
             (&mThreadParams.adiosAggregators)->default_value(0), "Number of aggregators [0 == number of MPI processes]")
            ("adios.ost", po::value<uint32_t > (&mThreadParams.adiosOST)->default_value(1),
             "Number of OST")
            ("adios.disable-meta", po::bool_switch (&mThreadParams.adiosDisableMeta)->default_value(false),
             "Disable online gather and write of a global meta file, can be time consuming (use `bpmeta` post-mortem)")
            ("adios.transport-params", po::value<std::string > (&mThreadParams.adiosTransportParams),
             "additional transport parameters, see ADIOS manual chapter 6.1.5, e.g., 'random_offset=1;stripe_count=4'")
            ("adios.compression", po::value<std::string >
             (&mThreadParams.adiosCompression)->default_value("none"),
             "ADIOS compression method, e.g., zlib (see `adios_config -m` for help)")
            ("adios.file", po::value<std::string > (&filename)->default_value(filename),
             "ADIOS output file")
            ("adios.checkpoint-file", po::value<std::string > (&checkpointFilename),
             "Optional ADIOS checkpoint filename (prefix)")
            ("adios.restart-file", po::value<std::string > (&restartFilename),
             "adios restart filename (prefix)")
            /* 50,000 particles are around 200 frames at 256 particles per frame (each 8k memory)
             * and match ~400MiB with typical picongpu particles.
             **/
            ("adios.restart-chunkSize", po::value<uint32_t > (&restartChunkSize)->default_value(50000),
             "Number of particles processed in one kernel call during restart to prevent frame count blowup");
    }

    std::string pluginGetName() const
    {
        return "ADIOSWriter";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    __host__ void notify(uint32_t currentStep)
    {
        notificationReceived(currentStep, false);
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        this->checkpointDirectory = checkpointDirectory;

        notificationReceived(currentStep, true);
    }

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        std::stringstream adiosPathBase;
        adiosPathBase << ADIOS_PATH_ROOT << restartStep << "/";
        mThreadParams.adiosBasePath = adiosPathBase.str();
        //mThreadParams.isCheckpoint = isCheckpoint;
        mThreadParams.currentStep = restartStep;
        mThreadParams.cellDescription = this->cellDescription;

        /** one could try ADIOS_READ_METHOD_BP_AGGREGATE too which might
         *  be beneficial for re-distribution on a different number of GPUs
         *    would need: - export chunk_size=<size> in MB
         *                - mpiTransportParams.c_str() in adios_read_init_method
         */
        ADIOS_CMD(adios_read_init_method(ADIOS_READ_METHOD_BP,
                                         mThreadParams.adiosComm,
                                         "verbose=3;abort_on_error;"));

        /* if restartFilename is relative, prepend with restartDirectory */
        if (!boost::filesystem::path(restartFilename).has_root_path())
        {
            restartFilename = restartDirectory + std::string("/") + restartFilename;
        }

        std::stringstream strFname;
        strFname << restartFilename << "_" << mThreadParams.currentStep << ".bp";

        // adios_read_open( fname, method, comm, lock_mode, timeout_sec )
        log<picLog::INPUT_OUTPUT > ("ADIOS: open file: %1%") % strFname.str();

        // when reading in BG_AGGREGATE mode, adios can not distinguish between
        // "file does not exist" and "stream is not (yet) available, so we
        // test it our selves
        if (!boost::filesystem::exists(strFname.str()))
            throw std::runtime_error("ADIOS: File does not exist.");

        /* <0 sec: wait forever
         * >=0 sec: return immediately if stream is not available */
        float_32 timeout = 0.0f;
        mThreadParams.fp = adios_read_open(strFname.str().c_str(),
                        ADIOS_READ_METHOD_BP, mThreadParams.adiosComm,
                        ADIOS_LOCKMODE_CURRENT, timeout);

        /* stream reading is tricky, see ADIOS manual section 8.11.1 */
        while (adios_errno == err_file_not_found)
        {
            /** \todo add c++11 platform independent sleep */
#if !defined(_WIN32)
            /* give the file system 1s of peace and quiet */
            usleep(1e6);
#endif
            mThreadParams.fp = adios_read_open(strFname.str().c_str(),
                        ADIOS_READ_METHOD_BP, mThreadParams.adiosComm,
                        ADIOS_LOCKMODE_CURRENT, timeout);
        }
        if (adios_errno == err_end_of_stream )
            /* could not read full stream */
            throw std::runtime_error("ADIOS: Stream terminated too early: " +
                                     std::string(adios_errmsg()) );
        if (mThreadParams.fp == NULL)
            throw std::runtime_error("ADIOS: Error opening stream: " +
                                     std::string(adios_errmsg()) );

        /* ADIOS types */
        AdiosUInt32Type adiosUInt32Type;

        /* load number of slides to initialize MovingWindow */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) read attr (%1% available)") %
            mThreadParams.fp->nattrs;
        void* slidesPtr = NULL;
        int slideSize;
        enum ADIOS_DATATYPES slidesType;
        ADIOS_CMD(adios_get_attr( mThreadParams.fp,
                                  (mThreadParams.adiosBasePath + std::string("sim_slides")).c_str(),
                                  &slidesType,
                                  &slideSize,
                                  &slidesPtr ));

        uint32_t slides = *( (uint32_t*)slidesPtr );
        log<picLog::INPUT_OUTPUT > ("ADIOS: value of sim_slides = %1%") %
            slides;

        PMACC_ASSERT(slidesType == adiosUInt32Type.type);
        PMACC_ASSERT(slideSize == sizeof(uint32_t)); // uint32_t in bytes

        void* lastStepPtr = NULL;
        int lastStepSize;
        enum ADIOS_DATATYPES lastStepType;
        ADIOS_CMD(adios_get_attr( mThreadParams.fp,
                                  (mThreadParams.adiosBasePath + std::string("iteration")).c_str(),
                                  &lastStepType,
                                  &lastStepSize,
                                  &lastStepPtr ));
        uint32_t lastStep = *( (uint32_t*)lastStepPtr );
        log<picLog::INPUT_OUTPUT > ("ADIOS: value of iteration = %1%") %
            lastStep;

        PMACC_ASSERT(lastStepType == adiosUInt32Type.type);
        PMACC_ASSERT(lastStep == restartStep);

        /* apply slides to set gpus to last/written configuration */
        log<picLog::INPUT_OUTPUT > ("ADIOS: Setting slide count for moving window to %1%") % slides;
        MovingWindow::getInstance().setSlideCounter(slides, restartStep);

        /* re-distribute the local offsets in y-direction
         * this will work for restarts with moving window still enabled
         * and restarts that disable the moving window
         * \warning enabling the moving window from a checkpoint that
         *          had no moving window will not work
         */
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        gc.setStateAfterSlides(slides);

        /* set window for restart, complete global domain */
        mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(restartStep);
        mThreadParams.localWindowToDomainOffset = DataSpace<simDim>::create(0);

        /* load all fields */
        ForEach<FileCheckpointFields, LoadFields<bmpl::_1> > forEachLoadFields;
        forEachLoadFields(&mThreadParams);

        /* load all particles */
        ForEach<FileCheckpointParticles, LoadSpecies<bmpl::_1> > forEachLoadSpecies;
        forEachLoadSpecies(&mThreadParams, restartChunkSize);

        IdProvider<simDim>::State idProvState;
        ReadNDScalars<uint64_t, uint64_t>()(mThreadParams,
                "picongpu/idProvider/startId", &idProvState.startId,
                "maxNumProc", &idProvState.maxNumProc);
        ReadNDScalars<uint64_t>()(mThreadParams,
                "picongpu/idProvider/nextId", &idProvState.nextId);
        log<picLog::INPUT_OUTPUT > ("Setting next free id on current rank: %1%") % idProvState.nextId;
        IdProvider<simDim>::setState(idProvState);

        /* free memory allocated in ADIOS calls */
        free(slidesPtr);
        free(lastStepPtr);

        /* clean shut down: close file and finalize */
        adios_release_step( mThreadParams.fp );
        ADIOS_CMD(adios_read_close( mThreadParams.fp ));
        ADIOS_CMD(adios_read_finalize_method(ADIOS_READ_METHOD_BP));
    }

private:

    void endAdios()
    {
        /* Finalize adios library */
        ADIOS_CMD(adios_finalize(Environment<simDim>::get().GridController()
                .getCommunicator().getRank()));

        __deleteArray(mThreadParams.fieldBfr);
    }

    void beginAdios(const std::string adiosFilename)
    {
        std::stringstream full_filename;
        full_filename << adiosFilename << "_" << mThreadParams.currentStep << ".bp";

        mThreadParams.fullFilename = full_filename.str();
        mThreadParams.adiosFileHandle = ADIOS_INVALID_HANDLE;

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
     * @param isCheckpoint checkpoint notification
     */
    void notificationReceived(uint32_t currentStep, bool isCheckpoint)
    {
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        mThreadParams.isCheckpoint = isCheckpoint;
        mThreadParams.currentStep = currentStep;
        mThreadParams.cellDescription = this->cellDescription;

        __getTransactionEvent().waitForFinished();

        std::string bpFilename( filename );
        std::string bpFiledir( outputDirectory );
        if( isCheckpoint )
        {
            bpFilename = checkpointFilename;
            bpFiledir = checkpointDirectory;
        }

        /* if file name is relative, prepend with common directory */
        if( boost::filesystem::path(bpFilename).has_root_path() )
            mThreadParams.adiosFilename = bpFilename;
        else
            mThreadParams.adiosFilename = bpFiledir + "/" + bpFilename;

        /* window selection */
        if( isCheckpoint )
            mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);
        else
            mThreadParams.window = MovingWindow::getInstance().getWindow(currentStep);

        for (uint32_t i = 0; i < simDim; ++i)
        {
            mThreadParams.localWindowToDomainOffset[i] = 0;
            if (mThreadParams.window.globalDimensions.offset[i] > localDomain.offset[i])
            {
                mThreadParams.localWindowToDomainOffset[i] =
                    mThreadParams.window.globalDimensions.offset[i] -
                    localDomain.offset[i];
            }
        }


        /* copy species only one time per timestep to the host */
        if( lastSpeciesSyncStep != currentStep )
        {
            DataConnector &dc = Environment<>::get().DataConnector();

            /* synchronizes the MallocMCBuffer to the host side */
            dc.getData<MallocMCBuffer> (MallocMCBuffer::getName());

            /* here we are copying all species to the host side since we
             * can not say at this point if this time step will need all of them
             * for sure (checkpoint) or just some user-defined species (dump)
             */
            ForEach<FileCheckpointParticles, CopySpeciesToHost<bmpl::_1> > copySpeciesToHost;
            copySpeciesToHost();
            lastSpeciesSyncStep = currentStep;
            dc.releaseData(MallocMCBuffer::getName());
        }

        beginAdios(mThreadParams.adiosFilename);

        writeAdios((void*) &mThreadParams, mpiTransportParams);

        endAdios();
    }

    void pluginLoad()
    {
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        /* It is important that we never change the mpi_pos after this point
         * because we get problems with the restart.
         * Otherwise we do not know which gpu must load the ghost parts around
         * the sliding window.
         */
        mpi_pos = gc.getPosition();
        mpi_size = gc.getGpuNodes();

        /* if number of aggregators is not set we use all mpi process as aggregator*/
        if( mThreadParams.adiosAggregators == 0 )
           mThreadParams.adiosAggregators=mpi_size.productOfComponents();

        if (notifyPeriod > 0)
        {
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

            /** create notify directory */
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(outputDirectory);
        }

        /* Initialize adios library */
        mThreadParams.adiosComm = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_dup(gc.getCommunicator().getMPIComm(), &(mThreadParams.adiosComm)));
        mThreadParams.adiosBufferInitialized = false;

        /* select MPI method, #OSTs and #aggregators */
        std::stringstream strMPITransportParams;
        strMPITransportParams << "num_aggregators=" << mThreadParams.adiosAggregators
                              << ";num_ost=" << mThreadParams.adiosOST;
        /* create meta file offline/post-mortem with bpmeta */
        if( mThreadParams.adiosDisableMeta )
            strMPITransportParams << ";have_metadata_file=0";
        /* additional, uncovered transport parameters, e.g.,
         * use system-defaults for striping per aggregated file */
        if( ! mThreadParams.adiosTransportParams.empty() )
            strMPITransportParams << ";" << mThreadParams.adiosTransportParams;

        mpiTransportParams = strMPITransportParams.str();

        if( restartFilename.empty() )
        {
            restartFilename = checkpointFilename;
        }

        loaded = true;
    }

    void pluginUnload()
    {
        if (notifyPeriod > 0)
        {
            if (mThreadParams.adiosComm != MPI_COMM_NULL)
            {
                MPI_CHECK(MPI_Comm_free(&(mThreadParams.adiosComm)));
            }
        }
    }

    template<typename ComponentType>
    static void writeField(ThreadParams *params, const uint32_t sizePtrType,
                           ADIOS_DATATYPES adiosType,
                           const uint32_t nComponents, const std::string name,
                           void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("ADIOS: write field: %1% %2% %3%") %
            name % nComponents % ptr;

        const bool fieldTypeCorrect( boost::is_same<ComponentType, float_X>::value );
        PMACC_CASSERT_MSG(Precision_mismatch_in_Field_Components__ADIOS,fieldTypeCorrect);

        /* data to describe source buffer */
        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_full = field_layout.getDataSpace();
        DataSpace<simDim> field_no_guard = params->window.localDimensions.size;
        DataSpace<simDim> field_guard = field_layout.getGuard() + params->localWindowToDomainOffset;

        /* write the actual field data */
        for (uint32_t d = 0; d < nComponents; d++)
        {
            const size_t plane_full_size = field_full[1] * field_full[0] * nComponents;
            const size_t plane_no_guard_size = field_no_guard[1] * field_no_guard[0];

            /* copy strided data from source to temporary buffer
             *
             * \todo use d1Access as in `include/plugins/hdf5/writer/Field.hpp`
             */
            const int maxZ = simDim == DIM3 ? field_no_guard[2] : 1;
            const int guardZ = simDim == DIM3 ? field_guard[2] : 0;
            for (int z = 0; z < maxZ; ++z)
            {
                for (int y = 0; y < field_no_guard[1]; ++y)
                {
                    const size_t base_index_src =
                                (z + guardZ) * plane_full_size +
                                (y + field_guard[1]) * field_full[0] * nComponents;

                    const size_t base_index_dst =
                                z * plane_no_guard_size +
                                y * field_no_guard[0];

                    for (int x = 0; x < field_no_guard[0]; ++x)
                    {
                        size_t index_src = base_index_src + (x + field_guard[0]) * nComponents + d;
                        size_t index_dst = base_index_dst + x;

                        params->fieldBfr[index_dst] = ((float_X*)ptr)[index_src];
                    }
                }
            }

            /* Write the actual field data. The id is on the front of the list. */
            if (params->adiosFieldVarIds.empty())
                throw std::runtime_error("Cannot write field (var id list is empty)");

            int64_t adiosFieldVarId = *(params->adiosFieldVarIds.begin());
            params->adiosFieldVarIds.pop_front();
            ADIOS_CMD(adios_write_byid(params->adiosFileHandle, adiosFieldVarId, params->fieldBfr));
        }
    }

    static void *writeAdios(void *p_args, std::string mpiTransportParams)
    {

        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);
        threadParams->adiosGroupSize = 0;

        /* y direction can be negative for first gpu */
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        DataSpace<simDim> particleOffset(localDomain.offset);
        particleOffset.y() -= threadParams->window.globalDimensions.offset.y();

        /* ADIOS API change:
         * <= 1.10.0: `enum ADIOS_FLAG`
         * >= 1.11.0: `enum ADIOS_STATISTICS_FLAG`
         */
#if ( ( ADIOS_VERSION_MAJOR * 100 + ADIOS_VERSION_MINOR ) >= 111 )
        ADIOS_STATISTICS_FLAG noStatistics = adios_stat_no;
#else
        ADIOS_FLAG noStatistics = adios_flag_no;
#endif

        /* create adios group for fields without statistics */
        ADIOS_CMD(adios_declare_group(&(threadParams->adiosGroupHandle),
                ADIOS_GROUP_NAME,
                (threadParams->adiosBasePath + std::string("iteration")).c_str(),
                noStatistics));

        /* select MPI method, #OSTs and #aggregators */
        ADIOS_CMD(adios_select_method(threadParams->adiosGroupHandle,
                  "MPI_AGGREGATE", mpiTransportParams.c_str(), ""));

        threadParams->fieldsOffsetDims = precisionCast<uint64_t>(localDomain.offset);

        /* write created variable values */
        for (uint32_t d = 0; d < simDim; ++d)
        {
            /* dimension 1 is y and is the direction of the moving window (if any) */
            if (1 == d)
            {
                uint64_t offset = std::max(0, localDomain.offset.y() -
                                              threadParams->window.globalDimensions.offset.y());
                threadParams->fieldsOffsetDims[d] = offset;
            }

            threadParams->fieldsSizeDims[d] = threadParams->window.localDimensions.size[d];
            threadParams->fieldsGlobalSizeDims[d] = threadParams->window.globalDimensions.size[d];
        }

        /* collect size information for each field to be written and define
         * field variables
         */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) collecting fields.");
        threadParams->adiosFieldVarIds.clear();
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointFields, CollectFieldsSizes<bmpl::_1> > forEachCollectFieldsSizes;
            forEachCollectFieldsSizes(threadParams);
        }
        else
        {
            ForEach<FileOutputFields, CollectFieldsSizes<bmpl::_1> > forEachCollectFieldsSizes;
            forEachCollectFieldsSizes(threadParams);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) collecting fields.");

        /* collect size information for all attributes of all species and define
         * particle variables
         */
        threadParams->adiosParticleAttrVarIds.clear();
        threadParams->adiosSpeciesIndexVarIds.clear();
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) counting particles.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointParticles, ADIOSCountParticles<bmpl::_1> > adiosCountParticles;
            adiosCountParticles(threadParams);
        }
        else
        {
            ForEach<FileOutputParticles, ADIOSCountParticles<bmpl::_1> > adiosCountParticles;
            adiosCountParticles(threadParams);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) counting particles.");

        PMACC_AUTO(idProviderState, IdProvider<simDim>::getState());
        WriteNDScalars<uint64_t, uint64_t> writeIdProviderStartId("picongpu/idProvider/startId", "maxNumProc");
        WriteNDScalars<uint64_t, uint64_t> writeIdProviderNextId("picongpu/idProvider/nextId");
        writeIdProviderStartId.prepare(*threadParams, idProviderState.maxNumProc);
        writeIdProviderNextId.prepare(*threadParams);

        /* allocate buffer in MB according to our current group size */
        /* `1 + mem` minimum 1 MiB that we can write attributes on empty GPUs */
        size_t writeBuffer_in_MiB=1+threadParams->adiosGroupSize / 1024 / 1024;
        /* value `1.1` is the secure factor if we miss to count some small buffers*/
        size_t buffer_mem=static_cast<size_t>(1.1 * static_cast<float_64>(writeBuffer_in_MiB));
        adios_set_max_buffer_size(buffer_mem);
        threadParams->adiosBufferInitialized = true;

        /* open adios file. all variables need to be defined at this point */
        log<picLog::INPUT_OUTPUT > ("ADIOS: open file: %1%") % threadParams->fullFilename;
        ADIOS_CMD(adios_open(&(threadParams->adiosFileHandle), ADIOS_GROUP_NAME,
                threadParams->fullFilename.c_str(), "w", threadParams->adiosComm));

        if (threadParams->adiosFileHandle == ADIOS_INVALID_HANDLE)
            throw std::runtime_error("ADIOS: Failed to open file.");

        /* attributes written here are pure meta data */
        WriteMeta writeMetaAttributes;
        writeMetaAttributes(threadParams);

        /* set adios group size (total size of all data to be written)
         * besides the number of bytes for variables, this call also
         * calculates the overhead of meta data
         */
        uint64_t adiosTotalSize;
        ADIOS_CMD(adios_group_size(threadParams->adiosFileHandle,
                threadParams->adiosGroupSize, &adiosTotalSize));

        /* write fields */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing fields.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointFields, GetFields<bmpl::_1> > forEachGetFields;
            forEachGetFields(threadParams);
        }
        else
        {
            ForEach<FileOutputFields, GetFields<bmpl::_1> > forEachGetFields;
            forEachGetFields(threadParams);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing fields.");

        /* print all particle species */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing particle species.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointParticles, WriteSpecies<bmpl::_1> > writeSpecies;
            writeSpecies(threadParams, particleOffset);
        }
        else
        {
            ForEach<FileOutputParticles, WriteSpecies<bmpl::_1> > writeSpecies;
            writeSpecies(threadParams, particleOffset);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing particle species.");

        log<picLog::INPUT_OUTPUT>("ADIOS: Writing IdProvider state (StartId: %1%, NextId: %2%, maxNumProc: %3%)")
                % idProviderState.startId % idProviderState.nextId % idProviderState.maxNumProc;
        writeIdProviderStartId(*threadParams, idProviderState.startId);
        writeIdProviderNextId(*threadParams, idProviderState.nextId);

        /* close adios file, most likely the actual write point */
        log<picLog::INPUT_OUTPUT > ("ADIOS: closing file: %1%") % threadParams->fullFilename;
        ADIOS_CMD(adios_close(threadParams->adiosFileHandle));

        /* avoid deadlock between not finished PMacc tasks and MPI_Barrier */
        __getTransactionEvent().waitForFinished();
        /*\todo: copied from adios example, we might not need this ? */
        MPI_CHECK(MPI_Barrier(threadParams->adiosComm));

        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyPeriod;
    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;
    std::string outputDirectory;
    std::string checkpointDirectory;

    /* select MPI method, #OSTs and #aggregators */
    std::string mpiTransportParams;

    uint32_t restartChunkSize;
    uint32_t lastSpeciesSyncStep;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;
};

} //namespace adios
} //namespace picongpu

