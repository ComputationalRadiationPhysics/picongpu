/* Copyright 2013-2026 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Sergei Bastrakov, Alexander Debus
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
#include "picongpu/fields/Fields.hpp"
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/fields/absorber/Thickness.hpp"
#include "picongpu/fields/absorber/pml/Parameters.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/traits/IsFieldDomainBound.hpp"
#include "picongpu/traits/IsFieldOutputOptional.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            namespace pml
            {
                //! Number of Cartesian PML layers (negative/positive per axis)
                inline constexpr uint32_t numPmlLayers = 2 * simDim;

                //! Additional node values for E or B in PML
                struct NodeValues
                {
                    /* The first letter corresponds to x, y, z field components,
                     * the second to transverse directions for the component
                     */
                    float_X xy, xz, yx, yz, zx, zy;

                    //! Number of components per node value
                    static constexpr int numComponents = 6;

                    /** Construct node values
                     *
                     * @param initialValue initial value for all components
                     */
                    HDINLINE NodeValues(float_X const initialValue = 0._X);

                    /** Construction for compatibility with pmacc vectors
                     *
                     * @param initialValue initial value for all components
                     */
                    HDINLINE static const NodeValues create(float_X const initialValue);

                    /** Element access for compatibility with pmacc vectors
                     *
                     * This is a utility for checkpointing and does not need a device
                     * version. For performance considerations does not check that the index
                     * is valid and relies on the components being stored in order, without
                     * padding.
                     *
                     * @param idx index less than 6
                     */
                    HDINLINE float_X& operator[](uint32_t const idx);

                    /** Const element access for compatibility with pmacc vectors
                     *
                     * This is a utility for checkpointing and does not need a device
                     * version. For performance considerations does not check that the index
                     * is valid and relies on the components being stored in order, without
                     * padding.
                     *
                     * @param idx index less than 6
                     */
                    HDINLINE float_X const& operator[](uint32_t const idx) const;
                };

                //! Slab geometry description
                struct SlabInfo
                {
                    pmacc::DataSpace<simDim> begin;
                    pmacc::DataSpace<simDim> end;
                };

                /** Data box type used for PML fields in kernels
                 *
                 * Stores PML fields in explicit slabs and provides access via
                 * a simDim-dimensional grid index.
                 *
                 * @tparam T_DataBox underlying ND data box type
                 */
                template<typename T_DataBox>
                class OuterLayerBox
                {
                public:
                    //! Underlying data box type
                    using DataBox = T_DataBox;

                    //! Element type
                    using ValueType = typename DataBox::ValueType;

                    //! Grid index type to be used for access
                    using Idx = pmacc::DataSpace<simDim>;

                    /** Create an outer layer box
                     *
                     * @param gridLayout grid layout, as for normal fields
                     * @param slabInfo view geometry for each slab
                     * @param slabBoxes underlying array of data boxes, preallocated per slab
                     *            the constructed OuterLayerBox does not own the box memory,
                     *            so can only be used before the box is reallocated
                     */
                    OuterLayerBox(
                        GridLayout<simDim> const& gridLayout,
                        std::array<SlabInfo, numPmlLayers> const& slabInfo,
                        std::array<DataBox, numPmlLayers> const& slabBoxes);

                    /** Constant element access by a simDim-dimensional index
                     *
                     * @param idx grid index
                     */
                    HDINLINE ValueType const& operator()(Idx const& idx) const;

                    /** Element access by a simDim-dimensional index
                     *
                     * @param idx grid index
                     */
                    HDINLINE ValueType& operator()(Idx const& idx);

                private:
                    /** Return a local simDim-dimensional ND data box index
                     *  for a given grid index with guard
                     *
                     * @param idxWithGuard grid index with guard
                     * @return local DataBox index of the respective slabIdx
                     */
                    HDINLINE std::tuple<Idx, uint32_t> getDataBoxIdx(Idx const& idxWithGuard) const;

                    //! A single Cartesian slab that is part of the outer layer box
                    class Layer
                    {
                    public:
                        /** Create a layer
                         *
                         * @param beginIdx first index
                         * @param endIdx index right after the last
                         */
                        HDINLINE Layer(Idx const& beginIdx = Idx::create(0), Idx const& endIdx = Idx::create(0));

                        /** Create a layer
                         *
                         * @param inputSlabInfo create layer from slab meta data
                         */
                        HDINLINE Layer(SlabInfo const& inputSlabInfo);

                        /** Check if the layer contains given index
                         *
                         * @param idx grid index without guard
                         */
                        HDINLINE bool contains(Idx const& idx) const;

                        //! Return local slab index for a point inside the slab
                        HDINLINE Idx getLocalIdx(Idx const& idx) const;

                    private:
                        //! First index of the layer
                        Idx beginIdx;

                        //! Size of the layer
                        Idx size;
                    };

                    //! Cartesian layers constituting the outer layer.
                    std::array<Layer, numPmlLayers> layers;

                    //! Slab data boxes, do not own memory
                    std::array<DataBox, numPmlLayers> slabBoxes;

                    //! Guard size
                    Idx const guardSize;
                };

                /** Base class for implementation inheritance in classes for the
                 *  electromagnetic fields in PML
                 *
                 * Stores field values on host and device and provides data synchronization
                 * between them.
                 *
                 * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
                 * ISimulationData.
                 */
                class Field
                    : public SimulationFieldHelper<MappingDesc>
                    , public ISimulationData
                {
                public:
                    //! Type of each field value
                    using ValueType = NodeValues;

                    //! Number of components of ValueType, for serialization
                    static constexpr int numComponents = NodeValues::numComponents;

                    //! Unit type of field components
                    using UnitValueType = pmacc::math::Vector<float_64, numComponents>;

                    /** Type of host-device buffer for field values
                     *
                     * Each slab is stored as a simDim-dimensional box.
                     */
                    using Buffer = pmacc::GridBuffer<ValueType, simDim>;

                    /** Type of data box for field values on host and device
                     *
                     * Data box for a slab buffer.
                     */
                    using DataBoxType = pmacc::DataBox<pmacc::PitchedBox<ValueType, simDim>>;

                    //! Data box type used for PML fields in kernels
                    using OuterLayerBoxType = OuterLayerBox<DataBoxType>;

                    //! Size of supercell
                    using SuperCellSize = MappingDesc::SuperCellSize;

                    /** Create a field
                     *
                     * @param cellDescription mapping for kernels
                     * @param globalThickness global PML thickness
                     */
                    HINLINE Field(MappingDesc const& cellDescription, Thickness const& globalThickness);

                    //! Number of slabs in this field representation
                    HINLINE static constexpr uint32_t getNumSlabs()
                    {
                        return numPmlLayers;
                    }

                    //! Get a reference to a slab host-device buffer for the field values
                    HINLINE Buffer& getGridBuffer(uint32_t slabIdx);

                    //! Get the grid layout of a slab
                    HINLINE pmacc::GridLayout<simDim> getGridLayout(uint32_t slabIdx);

                    //! Get the host data box for slab field values
                    HINLINE DataBoxType getHostDataBox(uint32_t slabIdx);

                    //! Get the device data box for slab field values
                    HINLINE DataBoxType getDeviceDataBox(uint32_t slabIdx);

                    //! Get slab begin index in local grid coordinates without guard
                    HINLINE pmacc::DataSpace<simDim> getSlabBegin(uint32_t slabIdx) const;

                    //! Get slab end index in local grid coordinates without guard
                    HINLINE pmacc::DataSpace<simDim> getSlabEnd(uint32_t slabIdx) const;

                    //! Get slab size in local grid coordinates
                    HINLINE pmacc::DataSpace<simDim> getSlabSize(uint32_t slabIdx) const;

                    //! Get active slab-view begin index in local grid coordinates without guard
                    HINLINE pmacc::DataSpace<simDim> getSlabViewBegin(uint32_t slabIdx) const;

                    //! Get active slab-view end index in local grid coordinates without guard
                    HINLINE pmacc::DataSpace<simDim> getSlabViewEnd(uint32_t slabIdx) const;

                    //! Get active slab-view size in local grid coordinates
                    HINLINE pmacc::DataSpace<simDim> getSlabViewSize(uint32_t slabIdx) const;

                    /** Set per-step slab view geometry for kernel access
                     *
                     * View geometry follows the PML layer exclusion logic z -> x -> y
                     * to avoid overlapping PMLs. To be symmetric with respect to x- and z-planes
                     * (moving window axis) exclusions, the y-planes are given the lowest priority.
                     */
                    HINLINE void setSlabViews(Thickness const& localThickness);

                    //! Get the device outer layer data box for the field values
                    HINLINE OuterLayerBoxType getDeviceOuterLayerBox();

                    /** Start asynchronous communication of field values
                     *
                     * @param serialEvent event to depend on
                     */
                    HINLINE virtual EventTask asyncCommunication(EventTask serialEvent);

                    /** Reset the host-device buffer for field values
                     *
                     * @param currentStep index of time iteration
                     */
                    HINLINE void reset(uint32_t currentStep) override;

                    //! Synchronize device data with host data
                    HINLINE void syncToDevice() override;

                    //! Synchronize host data with device data
                    HINLINE void synchronize() override;

                private:
                    //! Slab ids (DIM3: x-,x+,y-,y+,z-,z+; DIM2: x-,x+,y-,y+)
                    static constexpr uint32_t slabXNeg = 0u;
                    static constexpr uint32_t slabXPos = 1u;
                    static constexpr uint32_t slabYNeg = 2u;
                    static constexpr uint32_t slabYPos = 3u;
                    static constexpr uint32_t slabZNeg = 4u;
                    static constexpr uint32_t slabZPos = 5u;

                    //! Compute begin/end of all slabs
                    HINLINE void initializeSlabInfo();

                    //! Host-device slab buffers for field values
                    std::array<std::unique_ptr<Buffer>, numPmlLayers> slabData;

                    //! Allocation/storage geometry of each slab
                    std::array<SlabInfo, numPmlLayers> slabInfo;

                    //! Per-step active, kernel view slab geometry with overlap exclusion
                    std::array<SlabInfo, numPmlLayers> slabViewInfo;

                    //! Grid layout for normal (non-PML) fields
                    pmacc::GridLayout<simDim> gridLayout;

                    // PML global thickness
                    Thickness globalThickness;
                };

                //! Data box type used for PML fields in kernels
                using FieldBox = Field::OuterLayerBoxType;

                /** Representation of the additional electric field components in PML
                 *
                 * Stores field values on host and device and provides data synchronization
                 * between them.
                 *
                 * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
                 * ISimulationData.
                 */
                class FieldE : public Field
                {
                public:
                    /** Create a field
                     *
                     * @param cellDescription mapping for kernels
                     * @param globalThickness global PML thickness
                     */
                    HINLINE FieldE(MappingDesc const& cellDescription, Thickness const& globalThickness)
                        : Field(cellDescription, globalThickness)
                    {
                    }

                    //! Get id
                    HINLINE SimulationDataId getUniqueId()
                    {
                        return getName();
                    }

                    //! Get units of field components
                    HDINLINE static UnitValueType getUnit()
                    {
                        return UnitValueType::create(sim.unit.eField());
                    }

                    /** Get unit representation as powers of the 7 base measures
                     *
                     * Characterizing the record's unit in SI
                     * (length L, mass M, time T, electric current I,
                     *  thermodynamic temperature theta, amount of substance N,
                     *  luminous intensity J)
                     */
                    static std::vector<float_64> getUnitDimension()
                    {
                        return picongpu::FieldE::getUnitDimension();
                    }

                    //! Get text name
                    static std::string getName()
                    {
                        return "Convolutional PML E";
                    }
                };

                /** Representation of the additional magnetic field components in PML
                 *
                 * Stores field values on host and device and provides data synchronization
                 * between them.
                 *
                 * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
                 * ISimulationData.
                 */
                class FieldB : public Field
                {
                public:
                    /** Create a field
                     *
                     * @param cellDescription mapping for kernels
                     * @param globalThickness global PML thickness
                     */
                    HINLINE FieldB(MappingDesc const& cellDescription, Thickness const& globalThickness)
                        : Field(cellDescription, globalThickness)
                    {
                    }

                    //! Get id
                    HINLINE SimulationDataId getUniqueId()
                    {
                        return getName();
                    }

                    //! Get units of field components
                    static UnitValueType getUnit()
                    {
                        return UnitValueType::create(sim.unit.bField());
                    }

                    /** Get unit representation as powers of the 7 base measures
                     *
                     * Characterizing the record's unit in SI
                     * (length L, mass M, time T, electric current I,
                     *  thermodynamic temperature theta, amount of substance N,
                     *  luminous intensity J)
                     */
                    static std::vector<float_64> getUnitDimension()
                    {
                        return picongpu::FieldB::getUnitDimension();
                    }

                    //! Get text name
                    static std::string getName()
                    {
                        return "Convolutional PML B";
                    }
                };

            } // namespace pml
        } // namespace absorber
    } // namespace fields

    namespace traits
    {
        /** Field position traits for checkpointing
         *
         * PML fields do not fit well, for now just copy the normal fields.
         * Specialize only for Yee cell type, as this is the only one supported.
         */
        template<uint32_t T_dim>
        struct FieldPosition<fields::YeeCell, fields::absorber::pml::FieldE, T_dim>
            : FieldPosition<fields::YeeCell, FieldE, T_dim>
        {
        };

        /** Field position traits for checkpointing
         *
         * PML fields do not fit well, for now just copy the normal fields.
         * Specialize only for Yee cell type, as this is the only one supported.
         */
        template<uint32_t T_dim>
        struct FieldPosition<fields::YeeCell, fields::absorber::pml::FieldB, T_dim>
            : FieldPosition<fields::YeeCell, FieldB, T_dim>
        {
        };

        /** Field domain boundness trait for output and checkpointing:
         *  PML fields are stored on the simulation domain grid via slab chunks.
         */
        template<>
        struct IsFieldDomainBound<fields::absorber::pml::FieldE> : std::true_type
        {
        };

        /** Field domain boundness trait for output and checkpointing:
         *  PML fields are stored on the simulation domain grid via slab chunks.
         */
        template<>
        struct IsFieldDomainBound<fields::absorber::pml::FieldB> : std::true_type
        {
        };

        /** Field optional output trait for output and checkpointing:
         *  PML fields are optional, they are only instantiated for the PML absorber.
         */
        template<>
        struct IsFieldOutputOptional<fields::absorber::pml::FieldE> : std::true_type
        {
        };

        /** Field optional output trait for output and checkpointing:
         *  PML fields are optional, they are only instantiated for the PML absorber.
         */
        template<>
        struct IsFieldOutputOptional<fields::absorber::pml::FieldB> : std::true_type
        {
        };

    } // namespace traits
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        //! Node value traits for checkpointing
        template<>
        struct GetComponentsType<picongpu::fields::absorber::pml::NodeValues, false>
        {
            using type = picongpu::float_X;
        };

        //! Node value traits for checkpointing
        template<>
        struct GetNComponents<picongpu::fields::absorber::pml::NodeValues, false>
        {
            static constexpr uint32_t value = picongpu::fields::absorber::pml::NodeValues::numComponents;
        };

    } // namespace traits
} // namespace pmacc

#include "picongpu/fields/absorber/pml/Field.tpp"
