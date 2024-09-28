/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
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

#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
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

                    /** Get linear size of the outer layer box
                     *
                     * @param gridLayout grid layout, as for normal fields
                     * @param globalThickness global PML thickness
                     */
                    HINLINE int getOuterLayerBoxLinearSize(
                        GridLayout<simDim> const& gridLayout,
                        Thickness const& globalThickness)
                    {
                        // All sizes are without guard, since Pml is only on the internal area
                        auto const gridDataSpace = gridLayout.sizeWithoutGuardND();
                        auto const nonPmlDataSpace = gridDataSpace
                            - (globalThickness.getPositiveBorder() + globalThickness.getNegativeBorder());
                        auto const numGridCells = gridDataSpace.productOfComponents();
                        auto const numNonPmlCells = nonPmlDataSpace.productOfComponents();
                        return numGridCells - numNonPmlCells;
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
                    return box(getLinearIdx(idx));
                }

                template<typename T_Value>
                HDINLINE typename OuterLayerBox<T_Value>::ValueType& OuterLayerBox<T_Value>::operator()(Idx const& idx)
                {
                    return box(getLinearIdx(idx));
                }

                template<typename T_Value>
                HDINLINE int OuterLayerBox<T_Value>::getLinearIdx(Idx const& idxWithGuard) const
                {
                    /* Each PML layer provide a contiguous 1d index range.
                     * The resulting index is a sum of the baseIdx representing the total
                     * size of all previous layers and an index inside the current layer.
                     */
                    auto const idx = idxWithGuard - guardSize;
                    int currentLayerBeginIdx = 0;
                    int result = -1;
                    for(Layer const& layer : layers)
                        if(layer.contains(idx))
                        {
                            /* Note: here we could have returned the result directly,
                             * but chose to have a single return for potential
                             * performance gains on GPU.
                             * 'break' can be used too because the if condition will evaluate only for one layer to
                             * true but this would create 'stack frame' usage within GPU kernel.
                             */
                            result = currentLayerBeginIdx + layer.getLinearIdx(idx);
                        }
                        else
                            currentLayerBeginIdx += layer.getVolume();
                    return result;
                }

                template<typename T_Value>
                HDINLINE OuterLayerBox<T_Value>::Layer::Layer(Idx const& beginIdx, Idx const& endIdx)
                    : beginIdx{beginIdx}
                    , size{endIdx - beginIdx}
                    , volume{size.productOfComponents()}
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
                HDINLINE int OuterLayerBox<T_Value>::Layer::getVolume() const
                {
                    return volume;
                }

                template<typename T_Value>
                HDINLINE int OuterLayerBox<T_Value>::Layer::getLinearIdx(Idx const& idx) const
                {
                    // Convert to 3d zero-based index, for 2d keep .z( ) == 0
                    pmacc::DataSpace<3> zeroBasedIdx{0, 0, 0};
                    for(uint32_t dim = 0u; dim < simDim; dim++)
                        zeroBasedIdx[dim] = idx[dim] - beginIdx[dim];
                    return zeroBasedIdx.x() + zeroBasedIdx.y() * size.x() + zeroBasedIdx.z() * size.y() * size.x();
                }

                Field::Field(MappingDesc const& cellDescription, Thickness const& globalThickness)
                    : SimulationFieldHelper<MappingDesc>(cellDescription)
                    , gridLayout(cellDescription.getGridLayout())
                    , globalThickness(globalThickness)
                {
                    /* Create a simDim-dimentional buffer
                     * with size = linearSize x 1 [x 1 for 3d]
                     */
                    auto size = pmacc::DataSpace<simDim>::create(1);
                    size[0] = detail::getOuterLayerBoxLinearSize(gridLayout, globalThickness);
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
                    auto const boxWrapper1d
                        = pmacc::DataBoxDim1Access<DataBoxType>{getDeviceDataBox(), data->getGridLayout().sizeND()};
                    /* Note: the outer layer box type just provides access to data,
                     * it does not own or make copy of the data (nor is that required)
                     */
                    return OuterLayerBoxType{gridLayout, globalThickness, boxWrapper1d};
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
