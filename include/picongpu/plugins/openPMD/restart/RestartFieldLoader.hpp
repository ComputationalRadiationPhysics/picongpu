/* Copyright 2014-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/fields/absorber/pml/Field.hpp"
#    include "picongpu/plugins/misc/ComponentNames.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/simulation/control/MovingWindow.hpp"
#    include "picongpu/traits/IsFieldDomainBound.hpp"
#    include "picongpu/traits/IsFieldOutputOptional.hpp"

#    include <pmacc/Environment.hpp>
#    include <pmacc/communication/manager_common.hpp>
#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/dimensions/DataSpace.hpp>
#    include <pmacc/dimensions/GridLayout.hpp>
#    include <pmacc/particles/frame_types.hpp>
#    include <pmacc/types.hpp>

#    include <sstream>
#    include <stdexcept>
#    include <string>

#    include <openPMD/openPMD.hpp>

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
            static void loadFieldAtOffset(
                Data& field,
                uint32_t const numComponents,
                std::string const& objectName,
                ThreadParams* params,
                uint32_t const currentStep,
                DataSpace<simDim> const& domainOffset,
                DataSpace<simDim> const& localDomainSize,
                DataSpace<simDim> const& destinationOffset)
            {
                auto const name_lookup_tpl = plugins::misc::getComponentNames(numComponents);

                using ValueType = typename Data::ValueType;
                field.getHostBuffer().setValue(ValueType::create(0.0));
                if(localDomainSize.productOfComponents() == 0u)
                {
                    field.hostToDevice();
                    return;
                }

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Mesh& mesh = series.iterations[currentStep].open().meshes[objectName];

                auto destBox = field.getHostBuffer().getDataBox();
                for(uint32_t n = 0; n < numComponents; ++n)
                {
                    ::openPMD::RecordComponent rc
                        = numComponents > 1 ? mesh[name_lookup_tpl[n]] : mesh[::openPMD::RecordComponent::SCALAR];
                    ::openPMD::Offset start
                        = asStandardVector<DataSpace<simDim> const&, ::openPMD::Offset>(domainOffset);
                    ::openPMD::Extent count
                        = asStandardVector<DataSpace<simDim> const&, ::openPMD::Extent>(localDomainSize);

                    eventSystem::getTransactionEvent().waitForFinished();
                    std::shared_ptr<float_X> field_container = rc.loadChunk<float_X>(start, count);
                    mesh.seriesFlush();

                    int const elementCount = localDomainSize.productOfComponents();
#    pragma omp parallel for simd
                    for(int linearId = 0; linearId < elementCount; ++linearId)
                    {
                        auto destIdx = pmacc::math::mapToND(localDomainSize, linearId) + destinationOffset;
                        destBox(destIdx)[n] = field_container.get()[linearId];
                    }
                }
                field.hostToDevice();
                eventSystem::getTransactionEvent().waitForFinished();
            }

            template<class Data>
            static void loadField(
                Data& field,
                uint32_t const numComponents,
                std::string objectName,
                ThreadParams* params,
                uint32_t const currentStep,
                bool const isDomainBound)
            {
                log<picLog::INPUT_OUTPUT>("Begin loading field '%1%'") % objectName;

                auto const name_lookup_tpl = plugins::misc::getComponentNames(numComponents);
                DataSpace<simDim> const field_guard = field.getGridLayout().guardSizeND();

                pmacc::Selection<simDim> const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

                using ValueType = typename Data::ValueType;
                field.getHostBuffer().setValue(ValueType::create(0.0));

                ::pmacc::math::Vector<uint64_t, simDim> domain_offset = localDomain.offset;
                DataSpace<simDim> local_domain_size = params->window.localDimensions.size;
                bool useLinearIdxAsDestination = false;

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Mesh& mesh = series.iterations[currentStep].open().meshes[objectName];

                /* Patch for non-domain-bound fields
                 * This is an ugly fix to allow output of reduced 1d PML buffers
                 */
                if(!isDomainBound)
                {
                    auto const field_layout = field.getGridLayout();
                    auto const field_no_guard = field_layout.sizeWithoutGuardND();
                    auto const elementCount = field_no_guard.productOfComponents();
                    uint64_t pmlTotalSize = 0;

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
                    eventSystem::getTransactionEvent().waitForFinished();
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
                        pmlTotalSize += localSizes.at(2u * r);
                    }
                    log<picLog::INPUT_OUTPUT>("openPMD:  (end) collect PML sizes for %1%") % objectName;


                    auto const expectedExtent = [&pmlTotalSize]()
                    {
                        if constexpr(simDim == 3u)
                            return ::openPMD::Extent{1u, 1u, pmlTotalSize};
                        else
                            return ::openPMD::Extent{1u, pmlTotalSize};
                    }();

                    if(auto const& extentOnDisk = mesh.begin()->second.getExtent(); extentOnDisk != expectedExtent)
                    {
                        log<picLog::INPUT_OUTPUT>(
                            "openPMD:  Skip loading for PML fields. Expecting extent %1%, found extent %2% on disk. "
                            "This may happen when restarting with a different domain decomposition.")
                            % pmlTotalSize % extentOnDisk.at(simDim - 1u);
                        return;
                    }

                    domain_offset = DataSpace<simDim>::create(0);
                    domain_offset[0] = domainOffset;
                    local_domain_size = DataSpace<simDim>::create(1);
                    local_domain_size[0] = elementCount;
                    useLinearIdxAsDestination = true;
                }

                auto destBox = field.getHostBuffer().getDataBox();
                for(uint32_t n = 0; n < numComponents; ++n)
                {
                    // Read the subdomain which belongs to our mpi position.
                    // The total grid size must match the grid size of the stored
                    // data.
                    log<picLog::INPUT_OUTPUT>("openPMD: Read from domain: offset=%1% size=%2%") % domain_offset
                        % local_domain_size;
                    ::openPMD::RecordComponent rc
                        = numComponents > 1 ? mesh[name_lookup_tpl[n]] : mesh[::openPMD::RecordComponent::SCALAR];

                    log<picLog::INPUT_OUTPUT>("openPMD: Read from field '%1%'") % objectName;

                    ::openPMD::Offset start
                        = asStandardVector<::pmacc::math::Vector<uint64_t, simDim>&, ::openPMD::Offset>(domain_offset);
                    ::openPMD::Extent count
                        = asStandardVector<DataSpace<simDim>&, ::openPMD::Extent>(local_domain_size);

                    log<picLog::INPUT_OUTPUT>("openPMD: Allocate %1% elements")
                        % local_domain_size.productOfComponents();

                    // avoid deadlock between not finished pmacc tasks and mpi calls
                    // in openPMD backends
                    eventSystem::getTransactionEvent().waitForFinished();

                    /*
                     * @todo float_X should be some kind of gridBuffer's
                     *       GetComponentsType<ValueType>::type
                     */
                    std::shared_ptr<float_X> field_container = rc.loadChunk<float_X>(start, count);

                    /* start a blocking read of all scheduled variables */
                    mesh.seriesFlush();


                    int const elementCount = local_domain_size.productOfComponents();

#    pragma omp parallel for simd
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
                            destIdx = pmacc::math::mapToND(params->window.localDimensions.size, linearId);
                            /* jump over guard and local sliding window offset*/
                            destIdx += field_guard + params->localWindowToDomainOffset;
                        }

                        destBox(destIdx)[n] = field_container.get()[linearId];
                    }
                }

                field.hostToDevice();

                eventSystem::getTransactionEvent().waitForFinished();

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
            HINLINE void operator()(ThreadParams* params, uint32_t const restartStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                ThreadParams* tp = params;

                // Skip optional fields
                if(traits::IsFieldOutputOptional<T_Field>::value && !dc.hasId(T_Field::getName()))
                    return;

                /* load field without copying data to host */
                auto field = dc.get<T_Field>(T_Field::getName());

                /* load from openPMD */
                bool const isDomainBound = traits::IsFieldDomainBound<T_Field>::value;

                RestartFieldLoader::loadField(
                    field->getGridBuffer(),
                    (uint32_t) T_Field::numComponents,
                    T_Field::getName(),
                    tp,
                    restartStep,
                    isDomainBound);
            }
        };

        template<>
        struct LoadFields<fields::absorber::pml::FieldE>
        {
            HINLINE void operator()(ThreadParams* params, uint32_t const restartStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                if(traits::IsFieldOutputOptional<fields::absorber::pml::FieldE>::value
                   && !dc.hasId(fields::absorber::pml::FieldE::getName()))
                    return;
                auto field = dc.get<fields::absorber::pml::FieldE>(fields::absorber::pml::FieldE::getName());
                auto const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                for(uint32_t slabIdx = 0u; slabIdx < fields::absorber::pml::FieldE::getNumSlabs(); ++slabIdx)
                {
                    auto const slabName = fields::absorber::pml::FieldE::getName() + "_slab" + std::to_string(slabIdx);
                    auto const slabBegin = field->getSlabBegin(slabIdx);
                    auto const slabSize = field->getSlabSize(slabIdx);
                    RestartFieldLoader::loadFieldAtOffset(
                        field->getGridBuffer(slabIdx),
                        static_cast<uint32_t>(fields::absorber::pml::FieldE::numComponents),
                        slabName,
                        params,
                        restartStep,
                        localDomain.offset + slabBegin,
                        slabSize,
                        DataSpace<simDim>::create(0));
                }
            }
        };

        template<>
        struct LoadFields<fields::absorber::pml::FieldB>
        {
            HINLINE void operator()(ThreadParams* params, uint32_t const restartStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                if(traits::IsFieldOutputOptional<fields::absorber::pml::FieldB>::value
                   && !dc.hasId(fields::absorber::pml::FieldB::getName()))
                    return;
                auto field = dc.get<fields::absorber::pml::FieldB>(fields::absorber::pml::FieldB::getName());
                auto const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                for(uint32_t slabIdx = 0u; slabIdx < fields::absorber::pml::FieldB::getNumSlabs(); ++slabIdx)
                {
                    auto const slabName = fields::absorber::pml::FieldB::getName() + "_slab" + std::to_string(slabIdx);
                    auto const slabBegin = field->getSlabBegin(slabIdx);
                    auto const slabSize = field->getSlabSize(slabIdx);
                    RestartFieldLoader::loadFieldAtOffset(
                        field->getGridBuffer(slabIdx),
                        static_cast<uint32_t>(fields::absorber::pml::FieldB::numComponents),
                        slabName,
                        params,
                        restartStep,
                        localDomain.offset + slabBegin,
                        slabSize,
                        DataSpace<simDim>::create(0));
                }
            }
        };

        using namespace pmacc;

    } /* namespace openPMD */
} /* namespace picongpu */

#endif
