/* Copyright 2014-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *                     Benjamin Worpitz, Franz Poeschel
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

#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/plugins/misc/ComponentNames.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/IsFieldDomainBound.hpp"

#include <pmacc/communication/manager_common.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/particles/frame_types.hpp>
#include <pmacc/types.hpp>

#include <openPMD/openPMD.hpp>

#include <sstream>
#include <stdexcept>
#include <string>


namespace picongpu
{
    namespace openPMD
    {
        /**
         * Helper class for openPMD plugin to load fields from parallel openPMD
         * storages.
         */
        class RestartFieldLoader
        {
        public:
            template<class Data>
            static void loadField(
                Data& field,
                const uint32_t numComponents,
                std::string objectName,
                ThreadParams* params,
                bool const isDomainBound)
            {
                log<picLog::INPUT_OUTPUT>("Begin loading field '%1%'") % objectName;

                auto const name_lookup_tpl = plugins::misc::getComponentNames(numComponents);
                const DataSpace<simDim> field_guard = field.getGridLayout().getGuard();

                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

                using ValueType = typename Data::ValueType;
                field.getHostBuffer().setValue(ValueType::create(0.0));

                DataSpace<simDim> domain_offset = localDomain.offset;
                DataSpace<simDim> local_domain_size = params->window.localDimensions.size;
                bool useLinearIdxAsDestination = false;

                /* Patch for non-domain-bound fields
                 * This is an ugly fix to allow output of reduced 1d PML buffers
                 */
                if(!isDomainBound)
                {
                    auto const field_layout = params->gridLayout;
                    auto const field_no_guard = field_layout.getDataSpaceWithoutGuarding();
                    auto const elementCount = field_no_guard.productOfComponents();

                    /* Scan the PML buffer local size along all local domains
                     * This code is symmetric to one in Field::writeField()
                     */
                    log<picLog::INPUT_OUTPUT>("openPMD:  (begin) collect PML sizes for %1%") % objectName;
                    auto& gridController = Environment<simDim>::get().GridController();
                    auto const numRanks = uint64_t{gridController.getGlobalSize()};
                    /* Use domain position-based rank, not MPI rank, to be independent
                     * of the MPI rank assignment scheme
                     */
                    auto const rank = uint64_t{gridController.getScalarPosition()};
                    std::vector<uint64_t> localSizes(2 * numRanks, 0u);
                    uint64_t localSizeInfo[2] = {static_cast<uint64_t>(elementCount), rank};
                    __getTransactionEvent().waitForFinished();
                    MPI_CHECK(MPI_Allgather(
                        localSizeInfo,
                        2,
                        MPI_UINT64_T,
                        &(*localSizes.begin()),
                        2,
                        MPI_UINT64_T,
                        gridController.getCommunicator().getMPIComm()));
                    uint64_t domainOffset = 0;
                    for(uint64_t r = 0; r < numRanks; ++r)
                    {
                        if(localSizes.at(2u * r + 1u) < rank)
                            domainOffset += localSizes.at(2u * r);
                    }
                    log<picLog::INPUT_OUTPUT>("openPMD:  (end) collect PML sizes for %1%") % objectName;

                    domain_offset = DataSpace<simDim>::create(0);
                    domain_offset[0] = static_cast<int>(domainOffset);
                    local_domain_size = DataSpace<simDim>::create(1);
                    local_domain_size[0] = elementCount;
                    useLinearIdxAsDestination = true;
                }

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Container<::openPMD::Mesh>& meshes = series.iterations[params->currentStep].meshes;

                auto destBox = field.getHostBuffer().getDataBox();
                for(uint32_t n = 0; n < numComponents; ++n)
                {
                    // Read the subdomain which belongs to our mpi position.
                    // The total grid size must match the grid size of the stored
                    // data.
                    log<picLog::INPUT_OUTPUT>("openPMD: Read from domain: offset=%1% size=%2%") % domain_offset
                        % local_domain_size;
                    ::openPMD::RecordComponent rc = numComponents > 1
                        ? meshes[objectName][name_lookup_tpl[n]]
                        : meshes[objectName][::openPMD::RecordComponent::SCALAR];

                    log<picLog::INPUT_OUTPUT>("openPMD: Read from field '%1%'") % objectName;

                    auto ndim = rc.getDimensionality();
                    ::openPMD::Offset start = asStandardVector<DataSpace<simDim>&, ::openPMD::Offset>(domain_offset);
                    ::openPMD::Extent count
                        = asStandardVector<DataSpace<simDim>&, ::openPMD::Extent>(local_domain_size);

                    log<picLog::INPUT_OUTPUT>("openPMD: Allocate %1% elements")
                        % local_domain_size.productOfComponents();

                    // avoid deadlock between not finished pmacc tasks and mpi calls
                    // in openPMD backends
                    __getTransactionEvent().waitForFinished();

                    /*
                     * @todo float_X should be some kind of gridBuffer's
                     *       GetComponentsType<ValueType>::type
                     */
                    std::shared_ptr<float_X> field_container = rc.loadChunk<float_X>(start, count);

                    /* start a blocking read of all scheduled variables */
                    series.flush();


                    int const elementCount = local_domain_size.productOfComponents();

#pragma omp parallel for simd
                    for(int linearId = 0; linearId < elementCount; ++linearId)
                    {
                        DataSpace<simDim> destIdx;
                        if(useLinearIdxAsDestination)
                        {
                            destIdx[0] = linearId;
                        }
                        else
                        {
                            /* calculate index inside the moving window domain which
                             * is located on the local grid*/
                            destIdx = DataSpaceOperations<simDim>::map(params->window.localDimensions.size, linearId);
                            /* jump over guard and local sliding window offset*/
                            destIdx += field_guard + params->localWindowToDomainOffset;
                        }

                        destBox(destIdx)[n] = field_container.get()[linearId];
                    }
                }

                field.hostToDevice();

                __getTransactionEvent().waitForFinished();

                log<picLog::INPUT_OUTPUT>("openPMD: Read from domain: offset=%1% size=%2%") % domain_offset
                    % local_domain_size;
                log<picLog::INPUT_OUTPUT>("openPMD: Finished loading field '%1%'") % objectName;
            }
        };

        /**
         * Helper class for openPMDWriter (forEach operator) to load a field from
         * openPMD
         *
         * @tparam T_Field field class to load
         */
        template<typename T_Field>
        struct LoadFields
        {
        public:
            HDINLINE void operator()(ThreadParams* params)
            {
#ifndef __CUDA_ARCH__
                DataConnector& dc = Environment<>::get().DataConnector();
                ThreadParams* tp = params;

                /* load field without copying data to host */
                auto field = dc.get<T_Field>(T_Field::getName(), true);
                tp->gridLayout = field->getGridLayout();

                /* load from openPMD */
                bool const isDomainBound = traits::IsFieldDomainBound<T_Field>::value;
                RestartFieldLoader::loadField(
                    field->getGridBuffer(),
                    (uint32_t) T_Field::numComponents,
                    T_Field::getName(),
                    tp,
                    isDomainBound);

                dc.releaseData(T_Field::getName());
#endif
            }
        };

        using namespace pmacc;

    } /* namespace openPMD */
} /* namespace picongpu */
