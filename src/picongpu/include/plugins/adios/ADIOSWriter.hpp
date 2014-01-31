/**
 * Copyright 2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
#include <cassert>
#include <sstream>
#include <list>
#include <vector>

#include "types.h"
#include "simulation_types.hpp"
#include "plugins/adios/ADIOSWriter.def"

#include "particles/frame_types.hpp"

#include <adios.h>

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.def"
#include "fields/FieldTmp.hpp"
#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/particleToGrid/energyDensity.kernel"
#include "particles/operations/CountParticles.hpp" 

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/GridLayout.hpp"
#include "dataManagement/ISimulationIO.hpp"
#include "moduleSystem/ModuleConnector.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "dimensions/TVec.h"

#include "plugins/IPluginModule.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>

#include "plugins/adios/WriteSpecies.hpp"


namespace picongpu
{

namespace adios
{

using namespace PMacc;

namespace bmpl = boost::mpl;

namespace po = boost::program_options;

/**
 * Writes simulation data to adios files.
 * Implements the ISimulationIO interface.
 *
 * @param ElectronsBuffer class description for electrons
 * @param IonsBuffer class description for ions
 * @param simDim dimension of the simulation (2-3)
 */
class ADIOSWriter : public ISimulationIO, public IPluginModule
{
public:

    /* filter particles by global position*/
    typedef bmpl::vector< PositionFilter3D<> > usedFilters;
    typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;

private:

    /* fiter is a rule which describes which particles should be copyied to host*/
    MyParticleFilter filter;

    template<typename UnitType>
    static std::vector<double> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<double> tmp(numComponents);
        for (uint i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }

    /** Write calculated fields to adios file.
     *
     */
    template< typename T >
    struct GetFields
    {
    private:

        static std::vector<double> getUnit()
        {
            typedef typename T::UnitValueType UnitType;
            UnitType unit = T::getUnit();
            return createUnit(unit, T::numComponents);
        }

    public:

        HDINLINE void operator()(RefWrapper<ThreadParams*> params, const DomainInformation domInfo)
        {
#ifndef __CUDA_ARCH__
            DataConnector &dc = DataConnector::getInstance();

            T* field = &(dc.getData<T > (T::getCommTag()));
            params.get()->gridLayout = field->getGridLayout();

            writeField(params.get(),
                       sizeof(float),
                       domInfo,
                       T::numComponents,
                       T::getName(),
                       getUnit(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getCommTag());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     * and write them to adios.
     *
     * FieldTmp is calculated on device and than dumped to adios.
     */
    template< typename ThisSolver, typename ThisSpecies >
    struct GetFields<FieldTmpOperation<ThisSolver, ThisSpecies> >
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
        HDINLINE void operator()(RefWrapper<ThreadParams*> tparam, const DomainInformation domInfo)
        {
            this->operator_impl(tparam, domInfo);
        }
    private:
        typedef typename FieldTmp::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToADIOS<ComponentType>::type AdiosType;

        /** Create a name for the adios identifier.
         */
        template< typename Solver, typename Species >
        static std::string getName()
        {
            std::stringstream str;
            str << FieldTmp::getName<Solver>();
            str << "_";
            str << Species::FrameType::getName();
            return str.str();
        }

        /** Get the unit for the result from the solver*/
        template<typename Solver>
        static std::vector<double> getUnit()
        {
            typedef typename FieldTmp::UnitValueType UnitType;
            UnitType unit = FieldTmp::getUnit<Solver>();
            const uint32_t components = GetNComponents<ValueType>::value;
            return createUnit(unit, components);
        }

        HINLINE void operator_impl(RefWrapper<ThreadParams*> params, const DomainInformation domInfo)
        {
            DataConnector &dc = DataConnector::getInstance();

            /*## update field ##*/

            /*load FieldTmp without copy data to host*/
            FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FIELD_TMP, true));
            /*load particle without copy particle data to host*/
            ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(
                                                                 ThisSpecies::FrameType::CommunicationTag, true));

            fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
            /*run algorithm*/
            fieldTmp->computeValue < CORE + BORDER, ThisSolver > (*speciesTmp, params.get()->currentStep);

            EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(fieldTmpEvent);
            /* copy data to host that we can write same to disk*/
            fieldTmp->getGridBuffer().deviceToHost();
            dc.releaseData(ThisSpecies::FrameType::CommunicationTag);
            /*## finish update field ##*/

            const uint32_t components = GetNComponents<ValueType>::value;
            SplashType splashType;

            params.get()->gridLayout = fieldTmp->getGridLayout();
            /*write data to ADIOS file*/
            writeField(params.get(),
                       sizeof(float),
                       domInfo,
                       adiosType,
                       components,
                       getName<ThisSolver, ThisSpecies>(),
                       getUnit<ThisSolver>(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FIELD_TMP);

        }

    };

public:

    ADIOSWriter() :
    filename("simDataAdios"),
    notifyFrequency(0)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~ADIOSWriter()
    {

    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("adios.period", po::value<uint32_t > (&notifyFrequency)->default_value(0),
             "enable ADIOS IO [for each n-th step]")
            ("adios.file", po::value<std::string > (&filename)->default_value(filename),
             "ADIOS output file");
    }

    std::string moduleGetName() const
    {
        return "ADIOSWriter";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    __host__ void notify(uint32_t currentStep)
    {
        mThreadParams.currentStep = (int32_t) currentStep;
        mThreadParams.gridPosition = SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();
        mThreadParams.cellDescription = this->cellDescription;
        this->filter.setStatus(false);

        mThreadParams.window = MovingWindow::getInstance().getVirtualWindow(currentStep);

        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            //enable filters for sliding window and configurate position filter
            this->filter.setStatus(true);

            this->filter.setWindowPosition(mThreadParams.window.localOffset, mThreadParams.window.localSize);
        }

        __getTransactionEvent().waitForFinished();

        openAdiosFile();

        writeAdios((void*) &mThreadParams);

        closeAdiosFile();

    }

private:

    void closeAdiosFile()
    {
        if (mThreadParams.adiosHandle != ADIOS_INVALID_HANDLE)
        {
            log<picLog::INPUT_OUTPUT > ("ADIOS: closing file: %1%") % filename;
            adios_close(mThreadParams.adiosHandle);

            __deleteArray(mThreadParams.fieldBfr);
        }
    }

    void openAdiosFile()
    {
        GridController<simDim> &gc = GridController<simDim>::getInstance();

        std::stringstream full_filename;
        full_filename << filename << "_" << mThreadParams.currentStep << ".bp";

        // open adios file
        log<picLog::INPUT_OUTPUT > ("ADIOS open file: %1%") % filename;
        mThreadParams.adiosHandle = ADIOS_INVALID_HANDLE;
        adios_open(&(mThreadParams.adiosHandle), "fields", full_filename.str().c_str(),
            "w", gc.getCommunicator().getMPIComm());
        
        if (mThreadParams.adiosHandle == ADIOS_INVALID_HANDLE)
            throw std::runtime_error("Failed to open ADIOS file");

        mThreadParams.fieldBfr = new float[mThreadParams.window.localSize.productOfComponents()];
    }

    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            mThreadParams.adiosHandle = ADIOS_INVALID_HANDLE;
            mThreadParams.fieldBfr = NULL;
            mThreadParams.gridPosition =
                SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();

            GridController<simDim> &gc = GridController<simDim>::getInstance();
            /* It is important that we never change the mpi_pos after this point 
             * because we get problems with the restart.
             * Otherwise we do not know which gpu must load the ghost parts around
             * the sliding window.
             */
            mpi_pos = gc.getPosition();
            mpi_size = gc.getGpuNodes();

            DataConnector::getInstance().registerObserver(this, notifyFrequency);
            
            adios_init(xmlFilename.c_str(), gc.getCommunicator().getMPIComm());
        }

        loaded = true;
    }

    void moduleUnload()
    {
        if (notifyFrequency > 0)
        {
            adios_finalize(GridController<simDim>::getInstance().getCommunicator().getRank());
        }
    }

    static void writeField(ThreadParams *params, const uint32_t sizePtrType,
                           const DomainInformation domInfo,
                           const uint32_t nComponents, const std::string name,
                           std::vector<double> unit, void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("ADIOS: write field: %1% %2% %3%") %
            name % nComponents % ptr;

        std::vector<std::string> name_lookup;
        {
            const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};
            for (uint32_t d = 0; d < nComponents; d++)
                name_lookup.push_back(name_lookup_tpl[d]);
        }

        /*data to describe source buffer*/
        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_full = field_layout.getDataSpace();
        DataSpace<simDim> field_no_guard = domInfo.domainSize;
        DataSpace<simDim> field_guard = field_layout.getGuard() + domInfo.localDomainOffset;
        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset = DataSpace<simDim>(
                                                                0,
                                                                params->window.slides *
                                                                params->window.localFullSize.y(),
                                                                0);
        DataSpace<simDim> splashGlobalDomainOffset;
        DataSpace<simDim> splashGlobalOffsetFile;
        DataSpace<simDim> splashGlobalDomainSize;

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashGlobalOffsetFile[d] = domInfo.domainOffset[d];
            splashGlobalDomainOffset[d] = domInfo.globalDomainOffset[d] + globalSlideOffset[d];
            splashGlobalDomainSize[d] = domInfo.globalDomainSize[d];
        }
        
        splashGlobalOffsetFile[1] = std::max(0, domInfo.domainOffset[1] -
                domInfo.globalDomainOffset[1]);

        for (uint32_t d = 0; d < nComponents; d++)
        {
            std::stringstream datasetName;
            datasetName << "fields_" << name;
            if (nComponents > 1)
                datasetName << "_" << name_lookup.at(d);
            
            const size_t plane_full_size = field_full[1] * field_full[0] * nComponents;
            const size_t plane_no_guard_size = field_no_guard[1] * field_no_guard[0];

            // copy strided data from source to temporary buffer
            for (int z = 0; z < field_no_guard[2]; ++z)
                for (int y = 0; y < field_no_guard[1]; ++y)
                {
                    const size_t base_index_src =
                                (z + field_guard[2]) * plane_full_size +
                                (y + field_guard[1]) * field_full[0] * nComponents;

                    const size_t base_index_dst =
                                z * plane_no_guard_size +
                                y * field_no_guard[0];

                    for (int x = 0; x < field_no_guard[0]; ++x)
                    {
                        size_t index_src = base_index_src + (x + field_guard[0]) * nComponents + d;
                        size_t index_dst = base_index_dst + x;

                        params->fieldBfr[index_dst] = ((float*)ptr)[index_src];
                    }
                }

                // write the actual field data
                adios_write(params->adiosHandle, datasetName.str().c_str(), params->fieldBfr);
        }

    }

    static void *writeAdios(void *p_args)
    {

        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        /* write number of slides to timestep in adios file*/
        //uint32_t slides = threadParams->window.slides;

        /* build clean domain info (picongpu view) */
        DomainInformation domInfo;
        /* set global offset (from physical origin) to our first gpu data area*/
        domInfo.localDomainOffset = threadParams->window.localOffset;
        domInfo.globalDomainOffset = threadParams->window.globalSimulationOffset;
        domInfo.globalDomainSize = threadParams->window.globalWindowSize;
        domInfo.domainOffset = threadParams->gridPosition;
        /* change only the offset of the first gpu
         * localDomainOffset is only non zero for the gpus on top
         */
        domInfo.domainOffset += domInfo.localDomainOffset;
        domInfo.domainSize = threadParams->window.localSize;

        /* y direction can be negative for first gpu*/
        DataSpace<simDim> particleOffset(threadParams->gridPosition);
        particleOffset.y() -= threadParams->window.globalSimulationOffset.y();

        /*print all fields*/
        ForEach<FileOutputFields, GetFields<void> > forEachGetFields;
        forEachGetFields(ref(threadParams), domInfo);

        /*print all particle species*/
        /*log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing particle species.");
        ForEach<FileOutputParticles, WriteSpecies<void> > writeSpecies;
        writeSpecies(ref(threadParams), std::string(), domInfo, particleOffset);
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing particle species.");*/

        return NULL;

        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            /* data domain = domain inside the sliding window
             * ghost domain = domain under the data domain (is laying only on bottom gpus)
             * end of data domain is the beginning of the ghost domain
             */
            domInfo.globalDomainOffset.y() += domInfo.globalDomainSize.y();
            domInfo.domainOffset.y() = domInfo.globalDomainOffset.y();
            domInfo.domainSize = threadParams->window.localFullSize;
            domInfo.domainSize.y() -= threadParams->window.localSize.y();
            domInfo.globalDomainSize = threadParams->window.globalSimulationSize;
            domInfo.globalDomainSize.y() -= domInfo.globalDomainOffset.y();
            domInfo.localDomainOffset = DataSpace<simDim > ();
            /* only important for bottom gpus*/
            domInfo.localDomainOffset.y() = threadParams->window.localSize.y();

            particleOffset = threadParams->gridPosition;
            particleOffset.y() = -threadParams->window.localSize.y();

            if (threadParams->window.isBottom == false)
            {
                /* set size for all gpu to zero which are not bottom gpus*/
                domInfo.domainSize.y() = 0;
            }
            /* for restart we only need bottom ghosts for particles */
            log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing particle species bottom.");
            /* print all particle species */
            writeSpecies(ref(threadParams), std::string("_ghosts"), domInfo, particleOffset);
            log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing particle species bottom.");
        }
        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyFrequency;
    std::string filename;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;
};

} //namespace adios
} //namespace picongpu

