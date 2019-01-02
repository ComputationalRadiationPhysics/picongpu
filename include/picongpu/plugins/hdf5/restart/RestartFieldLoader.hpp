/* Copyright 2014-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include <pmacc/particles/frame_types.hpp>
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/simulationControl/MovingWindow.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>

#include <splash/splash.h>

#include <string>
#include <sstream>


namespace picongpu
{

namespace hdf5
{

/**
 * Helper class for HDF5Writer plugin to load fields from parallel libSplash files.
 */
class RestartFieldLoader
{
public:
    template<class Data>
    static void loadField(Data& field, const uint32_t numComponents, std::string objectName, ThreadParams *params)
    {
        log<picLog::INPUT_OUTPUT > ("Begin loading field '%1%'") % objectName;
        const DataSpace<simDim> field_guard = field.getGridLayout().getGuard();

        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(params->currentStep);
        const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        field.getHostBuffer().setValue(float3_X::create(0.0));

        const std::string name_lookup[] = {"x", "y", "z"};

        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        globalSlideOffset.y() = numSlides * localDomain.size.y();

        Dimensions domain_offset(0, 0, 0);
        for (uint32_t d = 0; d < simDim; ++d)
            domain_offset[d] = localDomain.offset[d] + globalSlideOffset[d];

        if (Environment<simDim>::get().GridController().getPosition().y() == 0)
            domain_offset[1] += params->window.globalDimensions.offset.y();

        Dimensions local_domain_size;
        for (uint32_t d = 0; d < simDim; ++d)
            local_domain_size[d] = params->window.localDimensions.size[d];

        // avoid deadlock between not finished pmacc tasks and mpi calls in splash/HDF5
        __getTransactionEvent().waitForFinished();

        auto destBox = field.getHostBuffer().getDataBox();
        for (uint32_t i = 0; i < numComponents; ++i)
        {
            // Read the subdomain which belongs to our mpi position.
            // The total grid size must match the grid size of the stored data.
            log<picLog::INPUT_OUTPUT > ("Read from domain: offset=%1% size=%2%") %
                domain_offset.toString() % local_domain_size.toString();
            DomainCollector::DomDataClass data_class;
            DataContainer *field_container =
                params->dataCollector->readDomain(params->currentStep,
                                           (std::string("fields/") + objectName +
                                            std::string("/") + name_lookup[i]).c_str(),
                                           Domain(domain_offset, local_domain_size),
                                           &data_class);

            int elementCount = params->window.localDimensions.size.productOfComponents();

            for (int linearId = 0; linearId < elementCount; ++linearId)
            {
                /* calculate index inside the moving window domain which is located on the local grid*/
                DataSpace<simDim> destIdx = DataSpaceOperations<simDim>::map(params->window.localDimensions.size, linearId);
                /* jump over guard and local sliding window offset*/
                destIdx += field_guard + params->localWindowToDomainOffset;

                destBox(destIdx)[i] = ((float_X*) (field_container->getIndex(0)->getData()))[linearId];
            }

            delete field_container;
        }

        field.hostToDevice();

        __getTransactionEvent().waitForFinished();

        log<picLog::INPUT_OUTPUT > ("Read from domain: offset=%1% size=%2%") %
            domain_offset.toString() % local_domain_size.toString();
        log<picLog::INPUT_OUTPUT > ("Finished loading field '%1%'") % objectName;
    }

    template<class Data>
    static void cloneField(Data& fieldDest, Data& fieldSrc, std::string objectName)
    {
        log<picLog::INPUT_OUTPUT > ("Begin cloning field '%1%'") % objectName;
        DataSpace<simDim> field_grid = fieldDest.getGridLayout().getDataSpace();

        size_t elements = field_grid.productOfComponents();
        float3_X *ptrDest = fieldDest.getHostBuffer().getDataBox().getPointer();
        float3_X *ptrSrc = fieldSrc.getHostBuffer().getDataBox().getPointer();

        for (size_t k = 0; k < elements; ++k)
        {
            ptrDest[k] = ptrSrc[k];
        }

        fieldDest.hostToDevice();

        __getTransactionEvent().waitForFinished();

        log<picLog::INPUT_OUTPUT > ("Finished cloning field '%1%'") % objectName;
    }
};

/**
 * Hepler class for HDF5Writer (forEach operator) to load a field from HDF5
 *
 * @tparam FieldType field class to load
 */
template< typename FieldType >
struct LoadFields
{
public:

    HDINLINE void operator()(ThreadParams* params)
    {
#ifndef __CUDA_ARCH__
        DataConnector &dc = Environment<>::get().DataConnector();
        ThreadParams *tp = params;

        /* load field without copying data to host */
        std::shared_ptr< FieldType > field = dc.get< FieldType >( FieldType::getName(), true );

        /* load from HDF5 */
        RestartFieldLoader::loadField(
                field->getGridBuffer(),
                (uint32_t)FieldType::numComponents,
                FieldType::getName(),
                tp);

        dc.releaseData( FieldType::getName() );
#endif
    }

};

using namespace pmacc;
using namespace splash;

} //namespace hdf5
} //namespace picongpu
