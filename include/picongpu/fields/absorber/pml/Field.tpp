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

                    //! Check whether there is at least one PML cell
                    HINLINE bool hasAllocatedPmlCells(Thickness const& globalThickness)
                    {
                        for(uint32_t dim = 0u; dim < simDim; ++dim)
                            if(globalThickness(dim, 0) > 0u || globalThickness(dim, 1) > 0u)
                                return true;
                        return false;
                    }

                    //! Get begin indices of pml region, without guard
                    HINLINE pmacc::DataSpace<simDim>
                    getGlobalPMLBegin(GridLayout<simDim> const& gridLayout, Thickness const& globalThickness)
                    {
                        auto const gridSize = gridLayout.sizeWithoutGuardND();
                        auto const hasPmlCells = hasAllocatedPmlCells(globalThickness);
                        auto begin = pmacc::DataSpace<simDim>::create(0);
                        for(uint32_t dim = 0u; dim < simDim; ++dim)
                        {
                            auto const negativeSize = globalThickness(dim, 0);
                            auto const positiveSize = globalThickness(dim, 1);
                            if(!hasPmlCells)
                                begin[dim] = 0;
                            else if(negativeSize > 0u)
                                begin[dim] = 0;
                            else if(positiveSize > 0u)
                                begin[dim] = gridSize[dim] - positiveSize;
                            else
                                begin[dim] = 0;
                        }
                        return begin;
                    }

                    //! Get end indices of pml region, without guard
                    HINLINE pmacc::DataSpace<simDim> getGlobalPMLEnd(
                        GridLayout<simDim> const& gridLayout,
                        Thickness const& globalThickness)
                    {
                        auto const gridSize = gridLayout.sizeWithoutGuardND();
                        auto const hasPmlCells = hasAllocatedPmlCells(globalThickness);
                        auto end = pmacc::DataSpace<simDim>::create(0);
                        for(uint32_t dim = 0u; dim < simDim; ++dim)
                        {
                            auto const negativeSize = globalThickness(dim, 0);
                            auto const positiveSize = globalThickness(dim, 1);
                            if(!hasPmlCells)
                                end[dim] = gridSize[dim];
                            else if(positiveSize > 0u)
                                end[dim] = gridSize[dim];
                            else if(negativeSize > 0u)
                                end[dim] = negativeSize;
                            else
                                end[dim] = gridSize[dim];
                        }
                        return end;
                    }

                    //! Check if a grid index without guard belongs to allocated PML area
                    HINLINE bool isInsideAllocatedPml(
                        pmacc::DataSpace<simDim> const& idx,
                        GridLayout<simDim> const& gridLayout,
                        Thickness const& globalThickness)
                    {
                        auto const negativeSize = globalThickness.getNegativeBorder();
                        auto const positiveSize = globalThickness.getPositiveBorder();
                        auto const gridSize = gridLayout.sizeWithoutGuardND();
                        auto const positiveBegin = gridSize - positiveSize;
                        for(uint32_t dim = 0u; dim < simDim; ++dim)
                            if((idx[dim] < negativeSize[dim]) || (idx[dim] >= positiveBegin[dim]))
                                return true;
                        return false;
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
                    // Here it is safe to call the const version
                    auto constThis = const_cast<NodeValues const*>(this);
                    return const_cast<float_X&>((*constThis)[idx]);
                }

                HDINLINE float_X const& NodeValues::operator[](uint32_t const idx) const
                {
                    return *(&xy + idx);
                }

                template<typename T_Value>
                OuterLayerBox<T_Value>::OuterLayerBox(
                    GridLayout<simDim> const& gridLayout,
                    Thickness const& globalThickness,
                    DataBox box)
                    : box(box)
                    , guardSize(gridLayout.guardSizeND())
                    , globalPMLBegin(detail::getGlobalPMLBegin(gridLayout, globalThickness))
                {
                    auto const negativeSize = globalThickness.getNegativeBorder();
                    auto const positiveSize = globalThickness.getPositiveBorder();
                    /* The region of interest is grid without guard,
                     * which consists of PML and internal area
                     */
                    auto const gridSize = gridLayout.sizeWithoutGuardND();
                    auto const positiveBegin = gridSize - positiveSize;

                    // Note: since this should compile for 2d, .z( ) can't be used
                    using detail::makeIdx;
                    int layerIdx = 0;

                    // Define standard values of Layer z-origin and negative z-size for 2D simulations.
                    auto positiveBeginZ = 0;
                    auto negativeSizeZ = 0;

                    if constexpr(simDim == DIM3)
                    {
                        auto const negativeZLayer
                            = Layer{makeIdx(0, 0, 0), makeIdx(gridSize[0], gridSize[1], negativeSize[2])};
                        layers[layerIdx++] = negativeZLayer;
                        auto const positiveZLayer
                            = Layer{makeIdx(0, 0, positiveBegin[2]), makeIdx(gridSize[0], gridSize[1], gridSize[2])};
                        layers[layerIdx++] = positiveZLayer;

                        positiveBeginZ = positiveBegin[2];
                        negativeSizeZ = negativeSize[2];
                    }

                    auto const negativeYLayer
                        = Layer{makeIdx(0, 0, negativeSizeZ), makeIdx(gridSize[0], negativeSize[1], positiveBeginZ)};
                    layers[layerIdx++] = negativeYLayer;
                    auto const positiveYLayer = Layer{
                        makeIdx(0, positiveBegin[1], negativeSizeZ),
                        makeIdx(gridSize[0], gridSize[1], positiveBeginZ)};
                    layers[layerIdx++] = positiveYLayer;

                    auto const negativeXLayer = Layer{
                        makeIdx(0, negativeSize[1], negativeSizeZ),
                        makeIdx(negativeSize[0], positiveBegin[1], positiveBeginZ)};
                    layers[layerIdx++] = negativeXLayer;
                    auto const positiveXLayer = Layer{
                        makeIdx(positiveBegin[0], negativeSize[1], negativeSizeZ),
                        makeIdx(gridSize[0], positiveBegin[1], positiveBeginZ)};
                    layers[layerIdx++] = positiveXLayer;
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::ValueType const& OuterLayerBox<T_Value>::operator()(
                    Idx const& idx) const
                {
                    return box(getDataBoxIdx(idx));
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::ValueType& OuterLayerBox<T_Value>::operator()(Idx const& idx)
                {
                    return box(getDataBoxIdx(idx));
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::Idx OuterLayerBox<T_Value>::getDataBoxIdx(
                    Idx const& idxWithGuard) const
                {
                    auto const idx = idxWithGuard - guardSize;
                    bool isInPml = false;
                    for(Layer const& layer : layers)
                        if(layer.contains(idx))
                            isInPml = true;
                    PMACC_DEVICE_ASSERT_MSG(isInPml, "PML index is outside of allocated psi area.");
                    return idx - globalPMLBegin;
                }

                template<typename T_Value>
                HDINLINE OuterLayerBox<T_Value>::Layer::Layer(Idx const& beginIdx, Idx const& endIdx)
                    : beginIdx{beginIdx}
                    , size{endIdx - beginIdx}
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

                Field::Field(MappingDesc const& cellDescription, Thickness const& globalThickness)
                    : SimulationFieldHelper<MappingDesc>(cellDescription)
                    , gridLayout(cellDescription.getGridLayout())
                    , globalThickness(globalThickness)
                    , globalPMLBegin(detail::getGlobalPMLBegin(gridLayout, globalThickness))
                {
                    auto const end = detail::getGlobalPMLEnd(gridLayout, globalThickness);
                    auto const size = end - globalPMLBegin;
                    auto const guardSize = pmacc::DataSpace<simDim>::create(0);
                    auto const layout = pmacc::GridLayout<simDim>(size, guardSize);
                    data = std::make_unique<Buffer>(layout);
                }

                Field::Buffer& Field::getGridBuffer()
                {
                    return *data;
                }

                pmacc::GridLayout<simDim> Field::getGridLayout()
                {
                    return data->getGridLayout();
                }

                Field::DataBoxType Field::getHostDataBox()
                {
                    return data->getHostBuffer().getDataBox();
                }

                Field::DataBoxType Field::getDeviceDataBox()
                {
                    return data->getDeviceBuffer().getDataBox();
                }

                Field::OuterLayerBoxType Field::getDeviceOuterLayerBox()
                {
                    return OuterLayerBoxType{gridLayout, globalThickness, getDeviceDataBox()};
                }

                EventTask Field::asyncCommunication(EventTask serialEvent)
                {
                    return data->asyncCommunication(serialEvent);
                }

                void Field::reset(uint32_t)
                {
                    data->getHostBuffer().reset(true);
                    data->getDeviceBuffer().reset(false);
                }

                void Field::syncToDevice()
                {
                    data->hostToDevice();
                }

                void Field::synchronize()
                {
                    data->deviceToHost();
                }

            } // namespace pml
        } // namespace absorber
    } // namespace fields
} // namespace picongpu
