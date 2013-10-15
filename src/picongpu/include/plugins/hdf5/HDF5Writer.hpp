/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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
#include "plugins/hdf5/HDF5Writer.def"

#include "particles/frame_types.hpp"

#include "splash.h"
#include "basetypes/ColTypeDim.hpp"
#include "basetypes/ColTypeFloat.hpp"
#include "basetypes/ColTypeDouble.hpp"
#include "basetypes/ColTypeInt.hpp"
#include "basetypes/ColTypeBool.hpp"
#include "basetypes/ColTypeInt3Array.hpp"
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


#include "plugins/hdf5/WriteSpecies.hpp"


namespace picongpu
{

namespace hdf5
{

using namespace PMacc;

using namespace DCollector;
namespace bmpl = boost::mpl;

namespace po = boost::program_options;

/**
 * Writes simulation data to hdf5 files using libSplash.
 * Implements the ISimulationIO interface.
 *
 * @param ElectronsBuffer class description for electrons
 * @param IonsBuffer class description for ions
 * @param simDim dimension of the simulation (2-3)
 */
class HDF5Writer : public ISimulationIO, public IPluginModule
{
public:

    typedef bmpl::vector< PositionFilter3D<> > usedFilters;
    typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;

private:

    MyParticleFilter filter;

    template<typename UnitType>
    static std::vector<double> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<double> tmp(numComponents);
        for (uint i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }

    /** Write calculated fields to HDF5 file.
     *
     */
    template< typename T >
    struct GetDCFields
    {
    private:

        static std::vector<double> getUnit()
        {
            typedef typename T::UnitValueType UnitType;
            UnitType unit = T::getUnit();
            return createUnit(unit, T::numComponents);
        }

    public:

        HDINLINE void operator()(RefWrapper<ThreadParams*> params)
        {
#ifndef __CUDA_ARCH__
            DCollector::ColTypeFloat ctFloat;

            DataConnector &dc = DataConnector::getInstance();

            T* field = &(dc.getData<T > (T::getCommTag()));
            params.get()->gridLayout = field->getGridLayout();

            writeField(params.get(),
                       ctFloat,
                       T::numComponents,
                       T::getName(),
                       getUnit(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getCommTag());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     * and write them to hdf5.
     *
     * FieldTmp is calculated on device and than dumped to HDF5.
     */
    template< typename ThisSolver, typename ThisSpecies >
    struct GetDCFields<FieldTmpOperation<ThisSolver, ThisSpecies> >
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
        HDINLINE void operator()(RefWrapper<ThreadParams*> tparam)
        {
            this->operator_impl(tparam);
        }
    private:
        typedef typename FieldTmp::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;
        

        /** Create a name for the hdf5 identifier.
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

        HINLINE void operator_impl(RefWrapper<ThreadParams*> params)
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
            SplashType spashType;

            params.get()->gridLayout = fieldTmp->getGridLayout();
            /*write data to HDF5 file*/
            writeField(params.get(),
                       spashType,
                       components,
                       getName<ThisSolver, ThisSpecies>(),
                       getUnit<ThisSolver>(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FIELD_TMP);

        }

    };

public:

    HDF5Writer() :
    filename("h5"),
    notifyFrequency(0),
    compression(false),
    continueFile(false)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~HDF5Writer()
    {

    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("hdf5.period", po::value<uint32_t > (&notifyFrequency)->default_value(0),
             "enable HDF5 IO [for each n-th step]")
            ("hdf5.file", po::value<std::string > (&filename)->default_value(filename),
             "HDF5 output file")
            ("hdf5.compression", po::value<bool > (&compression)->zero_tokens(),
             "enable HDF5 compression")
            ("hdf5.continue", po::value<bool > (&continueFile)->zero_tokens(),
             "continue existing HDF5 file instead of creating a new one");
    }

    std::string moduleGetName() const
    {
        return "HDF5Writer";
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

        openH5File();

        writeHDF5((void*) &mThreadParams);

        closeH5File();

    }

private:

    void closeH5File()
    {
        if (mThreadParams.dataCollector != NULL)
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector with file: %1%") % filename;
            mThreadParams.dataCollector->close();
            delete mThreadParams.dataCollector;
            mThreadParams.dataCollector = NULL;
        }
    }

    void openH5File()
    {
        const uint32_t maxOpenFilesPerNode = 4;
        mThreadParams.dataCollector = new DCollector::DomainCollector(maxOpenFilesPerNode);

        // set attributes for datacollector files
        DCollector::DataCollector::FileCreationAttr attr;
        attr.enableCompression = this->compression;

        if (continueFile)
            attr.fileAccType = DCollector::DataCollector::FAT_WRITE;
        else
            attr.fileAccType = DCollector::DataCollector::FAT_CREATE;


        attr.mpiPosition.set(0, 0, 0);
        attr.mpiSize.set(1, 1, 1);

        for (uint32_t i = 0; i < simDim; ++i)
        {
            attr.mpiPosition[i] = mpi_pos[i];
            attr.mpiSize[i] = mpi_size[i];
        }

        // open datacollector
        try
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % filename;
            mThreadParams.dataCollector->open(filename.c_str(), attr);
        }
        catch (DCollector::DCException e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }

        continueFile = true; //set continue for the next open

    }

    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            mThreadParams.gridPosition =
                SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();

            GridController<simDim> &gc = GridController<simDim>::getInstance();
            /* it is important that we never change the mpi_pos after this point 
             * because we get problems with the restart.
             * Otherwise we not know which gpu must load the ghost parts around
             * the sliding window
             */
            mpi_pos = gc.getPosition();
            mpi_size = gc.getGpuNodes();

            DataConnector::getInstance().registerObserver(this, notifyFrequency);
        }

        loaded = true;
    }

    void moduleUnload()
    {

    }

    static void writeField(ThreadParams *params, DCollector::CollectionType& colType,
                           const uint32_t dims, const std::string name,
                           std::vector<double> unit, void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("HDF5 write field: %1% %2% %3%") %
            name % dims % ptr;

        std::vector<std::string> name_lookup;
        {
            const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};
            for (uint32_t d = 0; d < dims; d++)
                name_lookup.push_back(name_lookup_tpl[d]);
        }

        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_full = field_layout.getDataSpace();
        DataSpace<simDim> field_no_guard = params->window.localSize;
        DataSpace<simDim> field_guard = field_layout.getGuard() + params->window.localOffset;

        DataSpace<simDim> sim_offset = params->gridPosition - params->window.globalSimulationOffset;
        DataSpace<simDim> global_sim_size = params->window.globalSimulationSize;

        /*simulation attributes for data*/
        DCollector::ColTypeDouble ctDouble;

        DCollector::Dimensions domain_offset(0, 0, 0);
        DCollector::Dimensions domain_size(1, 1, 1);

        ///\todo these might be deprecated !
        DCollector::Dimensions sim_size(0, 0, 0);
        DCollector::Dimensions sim_global_offset(0, 0, 0);
        DCollector::Dimensions sim_global_size(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            sim_size[d] = field_no_guard[d];
            sim_global_size[d] = global_sim_size[d];
            /*fields of first gpu in simulation are NULL point*/
            if (sim_offset[d] > 0)
            {
                sim_global_offset[d] = sim_offset[d];
                domain_offset[d] = sim_offset[d];
            }

            domain_size[d] = field_no_guard[d];
        }



        //only write data if we have data
        // if (field_no_guard.y() > 0)
        {
            for (uint32_t d = 0; d < dims; d++)
            {
                std::stringstream str;
                str << name;
                if (dims > 1)
                    str << "_" << name_lookup.at(d);

                params->dataCollector->writeDomain(params->currentStep, /* id == time step */
                                                   colType, /* data type */
                                                   simDim, /* NDims of the field data (scalar, vector, ...) */
                                                   /* source buffer, stride, data size, offset */
                                                   DCollector::Dimensions(field_full[0] * dims, field_full[1], field_full[2]),
                                                   DCollector::Dimensions(dims, 1, 1),
                                                   DCollector::Dimensions(field_no_guard[0], field_no_guard[1], field_no_guard[2]),
                                                   DCollector::Dimensions(field_guard[0] * dims + d, field_guard[1], field_guard[2]),
                                                   str.str().c_str(), /* data set name */
                                                   domain_offset, /* offset in global domain */
                                                   domain_size, /* local size */
                                                   Dimensions(0, 0, 0), /* \todo offset of the global domain */
                                                   sim_global_size, /* size of the global domain */
                                                   DomainCollector::GridType,
                                                   ptr);

                params->dataCollector->writeAttribute(params->currentStep,
                                                      DCollector::ColTypeDim(), str.str().c_str(), "sim_size",
                                                      sim_size.getPointer());
                params->dataCollector->writeAttribute(params->currentStep,
                                                      DCollector::ColTypeDim(), str.str().c_str(), "sim_global_offset",
                                                      sim_global_offset.getPointer());
                params->dataCollector->writeAttribute(params->currentStep,
                                                      ctDouble, str.str().c_str(), "sim_unit", &(unit.at(d)));
            }
        }
    }

    static void *writeHDF5(void *p_args)
    {

        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        /*print all fields*/
        ForEach<Hdf5OutputFields, GetDCFields<void> > forEachGetFields;
        forEachGetFields(ref(threadParams));

        // write fields
        /// \todo this should be a type trait
        DCollector::ColTypeFloat ctFloat;

        // write particles
        DataSpace<simDim> sim_offset =
            threadParams->gridPosition - threadParams->window.globalSimulationOffset;
        DataSpace<simDim> localOffset = threadParams->window.localOffset;
        DataSpace<simDim> localSize = threadParams->window.localSize;

        /*print all particle species*/
        log<picLog::INPUT_OUTPUT > ("HDF5 begin to write particle species.");
        ForEach<Hdf5OutputParticles, WriteSpecies<void> > writeSpecies;
        writeSpecies(ref(threadParams), std::string(), sim_offset, localOffset, localSize);
        log<picLog::INPUT_OUTPUT > ("HDF5 end to write particle species.");

        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            /* not needed, we don't use top for restart
            DataSpace<simDim> sim_offset = threadParams->gridPosition;
            DataSpace<simDim> localOffset;
            DataSpace<simDim> localSize = threadParams->window.localSize;
            localSize.y() = threadParams->window.globalSimulationOffset.y();
            sim_offset = threadParams->gridPosition;

            if (threadParams->window.isTop)
            {
              
                log<picLog::INPUT_OUTPUT > ("HDF5 begin to write particle species top.");
                writeSpecies(ref(threadParams), std::string("_top_"), sim_offset, localOffset, localSize);
                log<picLog::INPUT_OUTPUT > ("HDF5 end to write particle species top.");
            }
             */
            sim_offset = threadParams->gridPosition;
            sim_offset.y() += threadParams->window.localSize.y();
            localOffset = DataSpace<simDim > ();
            localOffset.y() = threadParams->window.localSize.y();
            localSize = threadParams->window.localFullSize;
            localSize.y() -= threadParams->window.localSize.y();

            if (threadParams->window.isBottom)
            {
                /*print all particle species*/
                log<picLog::INPUT_OUTPUT > ("HDF5 begin to write particle species bottom.");
                writeSpecies(ref(threadParams), std::string("_bottom_"), sim_offset, localOffset, localSize);
                log<picLog::INPUT_OUTPUT > ("HDF5 end to write particle species bottom.");
            }
        }
        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyFrequency;
    std::string filename;
    bool compression;
    bool continueFile;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;

};

} //namepsace hdf5
} //namepsace picongpu

