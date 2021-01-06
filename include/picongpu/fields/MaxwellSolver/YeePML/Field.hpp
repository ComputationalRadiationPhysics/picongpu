/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Sergei Bastrakov
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
#include "picongpu/fields/Fields.def"
#include "picongpu/fields/MaxwellSolver/YeePML/Parameters.hpp"
#include "picongpu/fields/cellType/Yee.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/traits/IsFieldDomainBound.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace yeePML
            {
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
                    float_X& operator[](uint32_t const idx);

                    /** Const element access for compatibility with pmacc vectors
                     *
                     * This is a utility for checkpointing and does not need a device
                     * version. For performance considerations does not check that the index
                     * is valid and relies on the components being stored in order, without
                     * padding.
                     *
                     * @param idx index less than 6
                     */
                    float_X const& operator[](uint32_t const idx) const;
                };

                /** Data box type used for PML fields in kernels
                 *
                 * Only stores data in the PML area using the given 1d data box.
                 * Access is provided via a simDim-dimensional index, same as for other
                 * grid values.
                 *
                 * @tparam T_DataBox1d underlying 1d data box type
                 */
                template<typename T_DataBox1d>
                class OuterLayerBox
                {
                public:
                    //! Underlying data box type
                    using DataBox = T_DataBox1d;

                    //! Element type
                    using ValueType = typename DataBox::ValueType;

                    //! Grid index type to be used for access
                    using Idx = pmacc::DataSpace<simDim>;

                    /** Create an outer layer box
                     *
                     * Only stores data in the PML area using the given 1d data box.
                     * Access is provided via a simDim-dimensional index, same as for other
                     * grid values.
                     *
                     * @param gridLayout grid layout, as for normal fields
                     * @param globalThickness global PML thickness
                     * @param box underlying data box, preallocated to fit all data
                     *            the constructed OuterLayerBox does not own the box memory,
                     *            so can only be used before the box is reallocated
                     */
                    OuterLayerBox(GridLayout<simDim> const& gridLayout, Thickness const& globalThickness, DataBox box);

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
                    /** Convert a simDim-dimensional index to a linear one
                     *
                     * @param idxWithGuard grid index with guard
                     */
                    HDINLINE int getLinearIdx(Idx const& idxWithGuard) const;

                    //! A single Cartesial layer that is part of the outer layer box
                    class Layer
                    {
                    public:
                        /** Create a layer
                         *
                         * @param beginIdx first index
                         * @param endIdx index right after the last
                         */
                        HDINLINE Layer(Idx const& beginIdx = Idx::create(0), Idx const& endIdx = Idx::create(0));

                        /** Check if the layer contains given index
                         *
                         * @param idx grid index without guard
                         */
                        HDINLINE bool contains(Idx const& idx) const;

                        //! Get the simDim-dimensional volume of the layer
                        HDINLINE int getVolume() const;

                        /** Get a linear index inside a layer
                         *
                         * Same as in pmacc::DataBox, x is minor and z is major.
                         *
                         * @param idx grid index without guard
                         */
                        HDINLINE int getLinearIdx(Idx const& idx) const;

                    private:
                        //! First index of the layer
                        Idx beginIdx;

                        //! Size of the layer
                        Idx size;

                        //! simDim-dimensional volume of the layer
                        int volume;
                    };

                    //! Number of layers: a positive and a negative one for each axis
                    static constexpr auto numLayers = 2 * simDim;

                    /** Cartesian layers constituting the outer layer
                     *
                     * The ordering inside the array is z-y-x for 3d and y-x for 2d.
                     * However, it should not be relevant since the layers do not intersect,
                     * and logically it represents a set of layers
                     */
                    Layer layers[numLayers];

                    //! Data box, does not own memory
                    DataBox box;

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
                     * The buffer is logically 1d, but technically multidimentional
                     * for easier coupling to output utilities.
                     */
                    using Buffer = pmacc::GridBuffer<ValueType, simDim>;

                    /** Type of data box for field values on host and device
                     *
                     * The data box is logically 1d, but technically multidimentional
                     * for easier coupling to output utilities.
                     */
                    using DataBoxType = pmacc::DataBox<pmacc::PitchedBox<ValueType, simDim>>;

                    //! Data box type used for PML fields in kernels
                    using OuterLayerBoxType = OuterLayerBox<pmacc::DataBoxDim1Access<DataBoxType>>;

                    //! Size of supercell
                    using SuperCellSize = MappingDesc::SuperCellSize;

                    /** Create a field
                     *
                     * @param cellDescription mapping for kernels
                     * @param globalThickness global PML thickness
                     */
                    HINLINE Field(MappingDesc const& cellDescription, Thickness const& globalThickness);

                    //! Get a reference to the host-device buffer for the field values
                    HINLINE Buffer& getGridBuffer();

                    //! Get the grid layout
                    HINLINE pmacc::GridLayout<simDim> getGridLayout();

                    //! Get the host data box for the field values
                    HINLINE DataBoxType getHostDataBox();

                    //! Get the device data box for the field values
                    HINLINE DataBoxType getDeviceDataBox();

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
                    //! Host-device buffer for field values
                    std::unique_ptr<Buffer> data;

                    //! Grid layout for normal (non-PML) fields
                    pmacc::GridLayout<simDim> gridLayout;

                    // PML global thickness
                    Thickness globalThickness;
                };

                //! Data box type used for PML fields in kernels
                using FieldBox = Field::OuterLayerBoxType;

                /** Representation of the additinal electric field components in PML
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
                        return UnitValueType::create(UNIT_EFIELD);
                    }

                    /** Get unit representation as powers of the 7 base measures
                     *
                     * Characterizing the record's unit in SI
                     * (length L, mass M, time T, electric current I,
                     *  thermodynamic temperature theta, amount of substance N,
                     *  luminous intensity J)
                     */
                    HINLINE static std::vector<float_64> getUnitDimension()
                    {
                        return picongpu::FieldE::getUnitDimension();
                    }

                    //! Get text name
                    HINLINE static std::string getName()
                    {
                        return "Convolutional PML E";
                    }
                };

                /** Representation of the additinal magnetic field components in PML
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
                    HDINLINE static UnitValueType getUnit()
                    {
                        return UnitValueType::create(UNIT_BFIELD);
                    }

                    /** Get unit representation as powers of the 7 base measures
                     *
                     * Characterizing the record's unit in SI
                     * (length L, mass M, time T, electric current I,
                     *  thermodynamic temperature theta, amount of substance N,
                     *  luminous intensity J)
                     */
                    HINLINE static std::vector<float_64> getUnitDimension()
                    {
                        return picongpu::FieldB::getUnitDimension();
                    }

                    //! Get text name
                    HINLINE static std::string getName()
                    {
                        return "Convolutional PML B";
                    }
                };

            } // namespace yeePML
        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Field position traits for checkpointing
         *
         * PML fields do not fit well, for now just copy the normal fields.
         * Specialize only for Yee cell type, as this is the only one supported.
         */
        template<uint32_t T_dim>
        struct FieldPosition<fields::cellType::Yee, fields::maxwellSolver::yeePML::FieldE, T_dim>
            : FieldPosition<fields::cellType::Yee, FieldE, T_dim>
        {
        };

        /** Field position traits for checkpointing
         *
         * PML fields do not fit well, for now just copy the normal fields.
         * Specialize only for Yee cell type, as this is the only one supported.
         */
        template<uint32_t T_dim>
        struct FieldPosition<fields::cellType::Yee, fields::maxwellSolver::yeePML::FieldB, T_dim>
            : FieldPosition<fields::cellType::Yee, FieldB, T_dim>
        {
        };

        /** Field domain boundness trait for output and checkpointing:
         *  PML fields are not domain-bound
         */
        template<>
        struct IsFieldDomainBound<fields::maxwellSolver::yeePML::FieldE> : std::false_type
        {
        };

        /** Field domain boundness trait for output and checkpointing:
         *  PML fields are not domain-bound
         */
        template<>
        struct IsFieldDomainBound<fields::maxwellSolver::yeePML::FieldB> : std::false_type
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
        struct GetComponentsType<picongpu::fields::maxwellSolver::yeePML::NodeValues, false>
        {
            typedef picongpu::float_X type;
        };

        //! Node value traits for checkpointing
        template<>
        struct GetNComponents<picongpu::fields::maxwellSolver::yeePML::NodeValues, false>
        {
            static constexpr uint32_t value = picongpu::fields::maxwellSolver::yeePML::NodeValues::numComponents;
        };

    } // namespace traits
} // namespace pmacc
