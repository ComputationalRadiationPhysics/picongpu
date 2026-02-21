/* Copyright 2013-2025 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Sergei Bastrakov,
 *                     Alexander Debus
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/absorber/pml/Field.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            namespace pml
            {
                namespace detail
                {
                    /** Construct an simDim-dimensional index out of 3 components.
                     *
                     * For 2d z is ignored
                     *
                     * @param x x component
                     * @param y y component
                     * @param z z component
                     */
                    HDINLINE pmacc::DataSpace<simDim> makeIdx(int const x, int const y, int const z)
                    {
                        auto const idx = pmacc::DataSpace<3>{x, y, z};
                        pmacc::DataSpace<simDim> result;
                        for(uint32_t dim = 0u; dim < simDim; dim++)
                            result[dim] = idx[dim];
                        return result;
                    }

                    //! Validate global thickness against local domain for slab allocation
                    HINLINE void checkGlobalThickness(GridLayout<simDim> const& gridLayout, Thickness const& globalThickness)
                    {
                        auto const gridSize = gridLayout.sizeWithoutGuardND();
                        for(uint32_t dim = 0u; dim < simDim; ++dim)
                        {
                            auto const negativeSize = globalThickness(dim, 0);
                            auto const positiveSize = globalThickness(dim, 1);
                            if(negativeSize > gridSize[dim] || positiveSize > gridSize[dim])
                            {
                                std::ostringstream msg;
                                msg << "Requested global PML thickness exceeds local domain in dimension " << dim
                                    << " (negative=" << negativeSize << ", positive=" << positiveSize
                                    << ", localSize=" << gridSize[dim] << ").";
                                throw std::out_of_range(msg.str());
                            }
                        }
                    }

                    //! Return volume of a slab size
                    HINLINE uint64_t getVolume(pmacc::DataSpace<simDim> const& size)
                    {
                        uint64_t result = 1u;
                        for(uint32_t dim = 0u; dim < simDim; ++dim)
                            result *= size[dim];
                        return result;
                    }

                    //! Return allocation size with a non-zero fallback
                    HINLINE pmacc::DataSpace<simDim> getAllocationSize(pmacc::DataSpace<simDim> const& size)
                    {
                        if(getVolume(size) == 0u)
                            return pmacc::DataSpace<simDim>::create(1);
                        return size;
                    }

                } // namespace detail

                HDINLINE NodeValues::NodeValues(float_X const initialValue /* = 0._X */)
                    : xy(initialValue)
                    , xz(initialValue)
                    , yx(initialValue)
                    , yz(initialValue)
                    , zx(initialValue)
                    , zy(initialValue)
                {
                }

                HDINLINE const NodeValues NodeValues::create(float_X const initialValue)
                {
                    return NodeValues{initialValue};
                }

                HDINLINE float_X& NodeValues::operator[](uint32_t const idx)
                {
                    return *(&xy + idx);
                }

                HDINLINE float_X const& NodeValues::operator[](uint32_t const idx) const
                {
                    return *(&xy + idx);
                }

                template<typename T_Value>
                OuterLayerBox<T_Value>::OuterLayerBox(
                    GridLayout<simDim> const& gridLayout,
                    Thickness const& globalThickness,
                    std::array<DataBox, numLayers> const& inputSlabBoxes)
                    : slabBoxes(inputSlabBoxes)
                    , guardSize(gridLayout.guardSizeND())
                {
                    auto const negativeSize = globalThickness.getNegativeBorder();
                    auto const positiveSize = globalThickness.getPositiveBorder();
                    auto const gridSize = gridLayout.sizeWithoutGuardND();

                    // Note: since this should compile for 2d, .z( ) can't be used
                    using detail::makeIdx;
                    layers[Field::slabXNeg] = Layer{
                        makeIdx(0, 0, 0),
                        makeIdx(negativeSize[0], gridSize[1], simDim == DIM3 ? gridSize[2] : 1),
                        Field::slabXNeg};
                    layers[Field::slabXPos] = Layer{
                        makeIdx(gridSize[0] - positiveSize[0], 0, 0),
                        makeIdx(gridSize[0], gridSize[1], simDim == DIM3 ? gridSize[2] : 1),
                        Field::slabXPos};
                    layers[Field::slabYNeg] = Layer{
                        makeIdx(0, 0, 0),
                        makeIdx(gridSize[0], negativeSize[1], simDim == DIM3 ? gridSize[2] : 1),
                        Field::slabYNeg};
                    layers[Field::slabYPos] = Layer{
                        makeIdx(0, gridSize[1] - positiveSize[1], 0),
                        makeIdx(gridSize[0], gridSize[1], simDim == DIM3 ? gridSize[2] : 1),
                        Field::slabYPos};
                    if constexpr(simDim == DIM3)
                    {
                        layers[Field::slabZNeg] = Layer{
                            makeIdx(0, 0, 0),
                            makeIdx(gridSize[0], gridSize[1], negativeSize[2]),
                            Field::slabZNeg};
                        layers[Field::slabZPos] = Layer{
                            makeIdx(0, 0, gridSize[2] - positiveSize[2]),
                            makeIdx(gridSize[0], gridSize[1], gridSize[2]),
                            Field::slabZPos};
                    }
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::ValueType const& OuterLayerBox<T_Value>::operator()(
                    Idx const& idx) const
                {
                    uint32_t slabIdx = 0u;
                    auto const localIdx = getDataBoxIdx(idx, slabIdx);
                    return slabBoxes[slabIdx](localIdx);
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::ValueType& OuterLayerBox<T_Value>::operator()(Idx const& idx)
                {
                    uint32_t slabIdx = 0u;
                    auto const localIdx = getDataBoxIdx(idx, slabIdx);
                    return slabBoxes[slabIdx](localIdx);
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::Idx OuterLayerBox<T_Value>::getDataBoxIdx(
                    Idx const& idxWithGuard,
                    uint32_t& slabIdx) const
                {
                    auto const idx = idxWithGuard - guardSize;
                    auto const mapFromLayer = [&](uint32_t const layerId) -> bool {
                        auto const& layer = layers[layerId];
                        if(layer.contains(idx))
                        {
                            slabIdx = layer.getSlabIdx();
                            return true;
                        }
                        return false;
                    };
                    if constexpr(simDim == DIM3)
                    {
                        if(mapFromLayer(Field::slabZNeg))
                            return layers[Field::slabZNeg].getLocalIdx(idx);
                        if(mapFromLayer(Field::slabZPos))
                            return layers[Field::slabZPos].getLocalIdx(idx);
                    }
                    if(mapFromLayer(Field::slabYNeg))
                        return layers[Field::slabYNeg].getLocalIdx(idx);
                    if(mapFromLayer(Field::slabYPos))
                        return layers[Field::slabYPos].getLocalIdx(idx);
                    if(mapFromLayer(Field::slabXNeg))
                        return layers[Field::slabXNeg].getLocalIdx(idx);
                    if(mapFromLayer(Field::slabXPos))
                        return layers[Field::slabXPos].getLocalIdx(idx);
                    PMACC_ASSERT_MSG(false, "PML index is outside of allocated PML slabs.");
                    return detail::makeIdx(0, 0, 0);
                }

                template<typename T_Value>
                HDINLINE OuterLayerBox<T_Value>::Layer::Layer(
                    Idx const& beginIdx,
                    Idx const& endIdx,
                    uint32_t const slabIdx)
                    : beginIdx{beginIdx}
                    , size{endIdx - beginIdx}
                    , slabIdx{slabIdx}
                {
                }

                template<typename T_Value>
                HDINLINE bool OuterLayerBox<T_Value>::Layer::contains(Idx const& idx) const
                {
                    for(uint32_t dim = 0u; dim < simDim; dim++)
                        if((idx[dim] < beginIdx[dim]) || (idx[dim] >= beginIdx[dim] + size[dim]))
                            return false;
                    return true;
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::Idx OuterLayerBox<T_Value>::Layer::getLocalIdx(
                    Idx const& idx) const
                {
                    return idx - beginIdx;
                }

                template<typename T_Value>
                HDINLINE uint32_t OuterLayerBox<T_Value>::Layer::getSlabIdx() const
                {
                    return slabIdx;
                }

                Field::Field(MappingDesc const& cellDescription, Thickness const& globalThickness)
                    : SimulationFieldHelper<MappingDesc>(cellDescription)
                    , gridLayout(cellDescription.getGridLayout())
                    , globalThickness(globalThickness)
                {
                    detail::checkGlobalThickness(gridLayout, globalThickness);
                    initializeSlabInfo();

                    auto const zeroGuard = pmacc::DataSpace<simDim>::create(0);
                    uint32_t slabIdx = 0u;
                    for(auto& slab : slabData)
                    {
                        auto const size = detail::getAllocationSize(slabInfo[slabIdx].end - slabInfo[slabIdx].begin);
                        slab = std::make_unique<Buffer>(pmacc::GridLayout<simDim>(size, zeroGuard));
                        ++slabIdx;
                    }
                }

                void Field::initializeSlabInfo()
                {
                    auto const gridSize = gridLayout.sizeWithoutGuardND();
                    auto const negativeSize = globalThickness.getNegativeBorder();
                    auto const positiveSize = globalThickness.getPositiveBorder();

                    slabInfo[slabXNeg] = SlabInfo{
                        detail::makeIdx(0, 0, 0),
                        detail::makeIdx(negativeSize[0], gridSize[1], simDim == DIM3 ? gridSize[2] : 1)};
                    slabInfo[slabXPos] = SlabInfo{
                        detail::makeIdx(gridSize[0] - positiveSize[0], 0, 0),
                        detail::makeIdx(gridSize[0], gridSize[1], simDim == DIM3 ? gridSize[2] : 1)};
                    slabInfo[slabYNeg] = SlabInfo{
                        detail::makeIdx(0, 0, 0),
                        detail::makeIdx(gridSize[0], negativeSize[1], simDim == DIM3 ? gridSize[2] : 1)};
                    slabInfo[slabYPos] = SlabInfo{
                        detail::makeIdx(0, gridSize[1] - positiveSize[1], 0),
                        detail::makeIdx(gridSize[0], gridSize[1], simDim == DIM3 ? gridSize[2] : 1)};
                    if constexpr(simDim == DIM3)
                    {
                        slabInfo[slabZNeg] = SlabInfo{
                            detail::makeIdx(0, 0, 0),
                            detail::makeIdx(gridSize[0], gridSize[1], negativeSize[2])};
                        slabInfo[slabZPos] = SlabInfo{
                            detail::makeIdx(0, 0, gridSize[2] - positiveSize[2]),
                            detail::makeIdx(gridSize[0], gridSize[1], gridSize[2])};
                    }
                }

                Field::Buffer& Field::getGridBuffer(uint32_t const slabIdx)
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return *slabData[slabIdx];
                }

                pmacc::GridLayout<simDim> Field::getGridLayout(uint32_t const slabIdx)
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return slabData[slabIdx]->getGridLayout();
                }

                Field::DataBoxType Field::getHostDataBox(uint32_t const slabIdx)
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return slabData[slabIdx]->getHostBuffer().getDataBox();
                }

                Field::DataBoxType Field::getDeviceDataBox(uint32_t const slabIdx)
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return slabData[slabIdx]->getDeviceBuffer().getDataBox();
                }

                pmacc::DataSpace<simDim> Field::getSlabBegin(uint32_t const slabIdx) const
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return slabInfo[slabIdx].begin;
                }

                pmacc::DataSpace<simDim> Field::getSlabEnd(uint32_t const slabIdx) const
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return slabInfo[slabIdx].end;
                }

                pmacc::DataSpace<simDim> Field::getSlabSize(uint32_t const slabIdx) const
                {
                    PMACC_ASSERT(slabIdx < numSlabs);
                    return slabInfo[slabIdx].end - slabInfo[slabIdx].begin;
                }

                Field::OuterLayerBoxType Field::getDeviceOuterLayerBox()
                {
                    std::array<DataBoxType, numSlabs> slabBoxes;
                    uint32_t slabIdx = 0u;
                    for(auto& slabBox : slabBoxes)
                    {
                        slabBox = getDeviceDataBox(slabIdx);
                        ++slabIdx;
                    }
                    return OuterLayerBoxType{gridLayout, globalThickness, slabBoxes};
                }

                EventTask Field::asyncCommunication(EventTask serialEvent)
                {
                    auto event = serialEvent;
                    for(auto& slab : slabData)
                        event += slab->asyncCommunication(event);
                    return event;
                }

                void Field::reset(uint32_t)
                {
                    for(auto& slab : slabData)
                    {
                        slab->getHostBuffer().reset(true);
                        slab->getDeviceBuffer().reset(false);
                    }
                }

                void Field::syncToDevice()
                {
                    for(auto& slab : slabData)
                        slab->hostToDevice();
                }

                void Field::synchronize()
                {
                    for(auto& slab : slabData)
                        slab->deviceToHost();
                }

            } // namespace pml
        } // namespace absorber
    } // namespace fields
} // namespace picongpu
