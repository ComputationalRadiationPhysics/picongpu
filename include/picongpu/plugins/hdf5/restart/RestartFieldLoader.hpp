/* Copyright 2014-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
#include "picongpu/plugins/misc/ComponentNames.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"
#include "picongpu/traits/IsFieldDomainBound.hpp"

#include <pmacc/communication/manager_common.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/Environment.hpp>

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
    static void loadField(
        Data& field,
        const uint32_t numComponents,
        std::string objectName,
        ThreadParams *params,
        const bool isDomainBound
    )
    {
        log<picLog::INPUT_OUTPUT > ("Begin loading field '%1%'") % objectName;
        const DataSpace<simDim> field_guard = field.getGridLayout().getGuard();

        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(params->currentStep);
        const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        using ValueType = typename Data::ValueType;
        field.getHostBuffer().setValue(ValueType::create(0.0));

        const auto componentNames = plugins::misc::getComponentNames( numComponents );

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
        int elementCount = params->window.localDimensions.size.productOfComponents();
        bool useLinearIdxAsDestination = false;

        /* Patch for non-domain-bound fields
         * This is an ugly fix to allow output of reduced 1d PML buffers
         */
        if( !isDomainBound )
        {
            auto const field_layout = params->gridLayout;
            auto const field_no_guard = field_layout.getDataSpaceWithoutGuarding();
            elementCount = field_no_guard.productOfComponents();
            // Number of elements on each local domain
            local_domain_size = Dimensions(
                elementCount,
                1,
                1
            );

            /* Scan the PML buffer local size along all local domains
             * This code is symmetric to one in Field::writeField()
             */
            log< picLog::INPUT_OUTPUT > ("HDF5:  (begin) collect PML sizes for %1%") % objectName;
            auto & gridController = Environment<simDim>::get().GridController();
            auto const numRanks = uint64_t{ gridController.getGlobalSize() };
            /* Use domain position-based rank, not MPI rank, to be independent
             * of the MPI rank assignment scheme
             */
            auto const rank = uint64_t{ gridController.getScalarPosition() };
            std::vector< uint64_t > localSizes( 2 * numRanks, 0u );
            uint64_t localSizeInfo[ 2 ] = {
                static_cast<uint64_t>( elementCount ),
                rank
            };
            __getTransactionEvent().waitForFinished();
            MPI_CHECK(MPI_Allgather(
                localSizeInfo, 2, MPI_UINT64_T,
                &( *localSizes.begin() ), 2, MPI_UINT64_T,
                gridController.getCommunicator().getMPIComm()
            ));
            uint64_t domainOffset = 0;
            for( uint64_t r = 0; r < numRanks; ++r )
            {
                if( localSizes.at( 2u * r + 1u ) < rank )
                    domainOffset += localSizes.at( 2u * r );
            }
            log< picLog::INPUT_OUTPUT > ("HDF5:  (end) collect PML sizes for %1%") % objectName;

            domain_offset = Dimensions(
                domainOffset,
                0,
                0
            );
            useLinearIdxAsDestination = true;
        }

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
                                            std::string("/") + componentNames[i]).c_str(),
                                           Domain(domain_offset, local_domain_size),
                                           &data_class);

            for (int linearId = 0; linearId < elementCount; ++linearId)
            {
                DataSpace<simDim> destIdx;
                if( useLinearIdxAsDestination )
                {
                    destIdx[ 0 ] = linearId;
                }
                else
                {
                    /* calculate index inside the moving window domain which is located on the local grid*/
                    destIdx = DataSpaceOperations<simDim>::map(params->window.localDimensions.size, linearId);
                    /* jump over guard and local sliding window offset*/
                    destIdx += field_guard + params->localWindowToDomainOffset;
                }
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
 * @tparam T_Field field class to load
 */
template< typename T_Field >
struct LoadFields
{
public:

    HDINLINE void operator()(ThreadParams* params)
    {
#ifndef __CUDA_ARCH__
        DataConnector &dc = Environment<>::get().DataConnector();
        ThreadParams *tp = params;

        /* load field without copying data to host */
        std::shared_ptr< T_Field > field = dc.get< T_Field >( T_Field::getName(), true );
        tp->gridLayout = field->getGridLayout();

        /* load from HDF5 */
        bool const isDomainBound = traits::IsFieldDomainBound< T_Field >::value;
        RestartFieldLoader::loadField(
            field->getGridBuffer(),
            static_cast< uint32_t >( T_Field::numComponents ),
            T_Field::getName(),
            tp,
            isDomainBound
        );

        dc.releaseData( T_Field::getName() );
#endif
    }

};

using namespace pmacc;
using namespace splash;

} //namespace hdf5
} //namespace picongpu
