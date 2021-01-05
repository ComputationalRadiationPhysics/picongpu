/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Marco Garten,
 *                     Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/mappings/kernel/MappingDescription.hpp"
#include "pmacc/mappings/kernel/ExchangeMapping.hpp"
#include "pmacc/fields/tasks/FieldFactory.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <boost/core/ignore_unused.hpp>


namespace pmacc
{
    namespace fields
    {
        namespace operations
        {
            /** Add field values from a received temporary buffer (exchange) to the local box (border)
             *
             * @tparam T_numWorkers number of workers
             */
            template<uint32_t T_numWorkers>
            struct KernelAddExchangeToBorder
            {
                /** add intermediate box to the border of the local box
                 *
                 * The `template< typename T> operator+( T const & rhs )` must be defined for
                 * the value type of exchangeBox and destBox.
                 *
                 * @tparam T_DestBox pmacc::DataBox, type of the local box
                 * @tparam T_ExchangeBox pmacc::ExchangeBox, type of the intermediate box
                 * @tparam T_Extent pmacc::DataSpace, type to describe n-dimensional sizes
                 * @tparam T_Mapping mapper functor type
                 *
                 * @param destBox box to a local field
                 * @param exchangeBox exchange box with guard data from the neighboring GPU
                 * @param exchangeSize dimensions of exchangeBox
                 * @param direction the direction of exchangeBox
                 * @param mapper functor to map a CUDA block to a supercell
                 */
                template<
                    typename T_DestBox,
                    typename T_ExchangeBox,
                    typename T_Extent,
                    typename T_Mapping,
                    typename T_Acc>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_DestBox& destBox,
                    T_ExchangeBox const& exchangeBox,
                    T_Extent const& exchangeSize,
                    T_Extent const& direction,
                    T_Mapping const& mapper) const
                {
                    using namespace mappings::threads;

                    using SuperCellSize = typename T_Mapping::SuperCellSize;

                    // number of cells in a superCell
                    constexpr uint32_t numCells = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    constexpr uint32_t numWorkers = T_numWorkers;
                    PMACC_CONSTEXPR_CAPTURE int dim = T_Mapping::Dim;

                    uint32_t const workerIdx = cupla::threadIdx(acc).x;

                    DataSpace<dim> const blockCell(
                        mapper.getSuperCellIndex(DataSpace<dim>(cupla::blockIdx(acc))) * SuperCellSize::toRT());

                    // origin in area from local GPU
                    DataSpace<dim> nullSourceCell(mapper.getSuperCellIndex(DataSpace<dim>()) * SuperCellSize::toRT());

                    auto const numGuardSuperCells = mapper.getGuardingSuperCells();

                    ForEachIdx<IdxConfig<numCells, numWorkers>>{
                        workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                        // cell index within the superCell
                        DataSpace<dim> const cellIdx
                            = DataSpaceOperations<dim>::template map<SuperCellSize>(linearIdx);
                        DataSpace<dim> targetCell(blockCell + cellIdx);
                        DataSpace<dim> sourceCell(targetCell - nullSourceCell);

                        // supercell offset relative to the guard origin (in cells)
                        DataSpace<dim> superCellOffsetInGuard(
                            (sourceCell / SuperCellSize::toRT()) * SuperCellSize::toRT());

                        /* defines if the virtual worker needs to add the value from
                         * the exchange box to the cell in the border
                         */
                        bool addValue = true;

                        for(uint32_t d = 0; d < dim; ++d)
                        {
                            if(direction[d] == 1)
                            {
                                if(superCellOffsetInGuard[d] + cellIdx[d]
                                   < numGuardSuperCells[d] * SuperCellSize::toRT()[d] - exchangeSize[d])
                                    addValue = false;
                                sourceCell[d] -= numGuardSuperCells[d] * SuperCellSize::toRT()[d] - exchangeSize[d];
                                targetCell[d] -= numGuardSuperCells[d] * SuperCellSize::toRT()[d];
                            }
                            else if(direction[d] == -1)
                            {
                                if(superCellOffsetInGuard[d] + cellIdx[d] >= exchangeSize[d])
                                    addValue = false;
                                targetCell[d] += numGuardSuperCells[d] * SuperCellSize::toRT()[d];
                            }
                        }
                        if(addValue)
                            destBox(targetCell) += exchangeBox(sourceCell);
                    });
                }
            };


            /** add a exchange buffer to the border of the local buffer
             *
             * CopyGuardToExchange is the opposite operation for the neighboring
             * device to create an exchange which can be added with this functor.
             */
            struct AddExchangeToBorder
            {
                /** add exchange to border of the local buffer
                 *
                 * Add data cell-wise from the exchange to the border of the local buffer.
                 * The `template< typename T> operator+( T const & rhs )` must be defined for
                 * the value type of the buffer.
                 *
                 * @tparam T_DestBuffer pmacc::GridBuffer, type of the used buffer
                 * @tparam T_SuperCellSize pmacc::math::CT::vector, size of the supercell in each direction
                 *
                 * @param destBuffer destination buffer with exchanges
                 * @param superCellSize compile time supercell size
                 * @param exchangeType the exchange direction which needs to be copied
                 */
                template<typename T_DestBuffer, typename T_SuperCellSize>
                void operator()(
                    T_DestBuffer& destBuffer,
                    T_SuperCellSize const& superCellSize,
                    uint32_t const exchangeType) const
                {
                    boost::ignore_unused(superCellSize);

                    using SuperCellSize = T_SuperCellSize;

                    constexpr int dim = T_SuperCellSize::dim;

                    using MappingDesc = MappingDescription<dim, SuperCellSize>;

                    /* use only the x dimension to determine the number of supercells in the GUARD
                     *
                     * @warning pmacc restriction: all dimension must have the some number of guarding
                     * supercells
                     */
                    auto const numGuardSuperCells = destBuffer.getGridLayout().getGuard() / SuperCellSize::toRT();

                    MappingDesc const mappingDesc(destBuffer.getGridLayout().getDataSpace(), numGuardSuperCells);

                    ExchangeMapping<GUARD, MappingDesc> mapper(mappingDesc, exchangeType);

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    const DataSpace<dim> direction = Mask::getRelativeDirections<dim>(mapper.getExchangeType());

                    PMACC_KERNEL(KernelAddExchangeToBorder<numWorkers>{})
                    (mapper.getGridDim(), numWorkers)(
                        destBuffer.getDeviceBuffer().getDataBox(),
                        destBuffer.getReceiveExchange(exchangeType).getDeviceBuffer().getDataBox(),
                        destBuffer.getReceiveExchange(exchangeType).getDeviceBuffer().getDataSpace(),
                        direction,
                        mapper);
                }
            };

        } // namespace operations
    } // namespace fields
} // namespace pmacc
