/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
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
#include "picongpu/fields/cellType/Yee.hpp"
#include "picongpu/traits/FieldPosition.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/math/Vector.hpp>

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
        HDINLINE NodeValues( float_X const initialValue = 0._X ):
            xy( initialValue ),
            xz( initialValue ),
            yx( initialValue ),
            yz( initialValue ),
            zx( initialValue ),
            zy( initialValue )
        {
        }

        /** Construction for compatibility with pmacc vectors
         *
         * @param initialValue initial value for all components
         */
        HDINLINE static const NodeValues create( float_X const initialValue )
        {
            return NodeValues{ initialValue };
        }

        /** Element access for compatibility with pmacc vectors
         *
         * This is a utility for checkpointing and does not need a device
         * version. For performance considerations does not check that the index
         * is valid and relies on the components being stored in order, without
         * padding.
         *
         * @param idx index less than 6
         */
        float_X & operator[ ]( uint32_t const idx )
        {
            // Here it is safe to call the const version
            auto constThis = const_cast< NodeValues const * >( this );
            return const_cast< float_X & >( ( *constThis )[ idx ] );
        }

        /** Const element access for compatibility with pmacc vectors
         *
         * This is a utility for checkpointing and does not need a device
         * version. For performance considerations does not check that the index
         * is valid and relies on the components being stored in order, without
         * padding.
         *
         * @param idx index less than 6
         */
        float_X const & operator[ ]( uint32_t const idx ) const
        {
            return *( &xy + idx );
        }

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
    class Field : public SimulationFieldHelper< MappingDesc >, public ISimulationData
    {
    public:

        //! Type of each field value
        using ValueType = NodeValues;

        //! Number of components of ValueType, for serialization
        static constexpr int numComponents = NodeValues::numComponents;

        //! Unit type of field components
        using UnitValueType = pmacc::math::Vector< float_64, numComponents >;

        //! Type of data box for field values on host and device
        using DataBoxType = DataBox< PitchedBox< ValueType, simDim > >;

        //! Size of supercell
        using SuperCellSize = MappingDesc::SuperCellSize ;

        /** Create a field
         *
         * @param cellDescription mapping for kernels
         */
        HINLINE Field( MappingDesc const & cellDescription );

        //! Get a reference to the host-device buffer for the field values
        HINLINE GridBuffer< ValueType, simDim > & getGridBuffer( );

        //! Get the grid layout
        HINLINE GridLayout< simDim > getGridLayout( );

        //! Get the host data box for the field values
        HINLINE DataBoxType getHostDataBox( );

        //! Get the device data box for the field values
        HINLINE DataBoxType getDeviceDataBox( );

        /** Start asynchronous communication of field values
         *
         * @param serialEvent event to depend on
         */
        HINLINE virtual EventTask asyncCommunication( EventTask serialEvent );

        /** Reset the host-device buffer for field values
         *
         * @param currentStep index of time iteration
         */
        HINLINE void reset( uint32_t currentStep ) override;

        //! Synchronize device data with host data
        HINLINE void syncToDevice( ) override;

        //! Synchronize host data with device data
        HINLINE void synchronize( ) override;

    private:

        //! Type of host-device buffer for field values
        using Buffer = pmacc::GridBuffer<
            ValueType,
            simDim
        >;

        //! Host-device buffer for field values
        std::unique_ptr< Buffer > data;

    };

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
         */
        HINLINE FieldE( MappingDesc const & cellDescription ):
            Field( cellDescription )
        {
        }

        //! Get id
        HINLINE SimulationDataId getUniqueId( )
        {
            return getName( );
        }

        //! Get units of field components
        HDINLINE static UnitValueType getUnit( )
        {
            return UnitValueType::create( UNIT_EFIELD );
        }

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */
        HINLINE static std::vector< float_64 > getUnitDimension( )
        {
            return picongpu::FieldE::getUnitDimension( );
        }

        //! Get text name
        HINLINE static std::string getName( )
        {
            return "PML E components";
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
         */
        HINLINE FieldB( MappingDesc const & cellDescription ):
            Field( cellDescription )
        {
        }

        //! Get id
        HINLINE SimulationDataId getUniqueId( )
        {
            return getName( );
        }

        //! Get units of field components
        HDINLINE static UnitValueType getUnit( )
        {
            return UnitValueType::create( UNIT_BFIELD );
        }

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */
        HINLINE static std::vector< float_64 > getUnitDimension( )
        {
            return picongpu::FieldB::getUnitDimension( );
        }

        //! Get text name
        HINLINE static std::string getName( )
        {
            return "PML B components";
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
    template< uint32_t T_dim >
    struct FieldPosition<
        fields::cellType::Yee,
        fields::maxwellSolver::yeePML::FieldE,
        T_dim
    > : FieldPosition<
        fields::cellType::Yee,
        FieldE,
        T_dim
    >
    {
    };

    /** Field position traits for checkpointing
     *
     * PML fields do not fit well, for now just copy the normal fields.
     * Specialize only for Yee cell type, as this is the only one supported.
     */
    template< uint32_t T_dim >
    struct FieldPosition<
        fields::cellType::Yee,
        fields::maxwellSolver::yeePML::FieldB,
        T_dim
    > : FieldPosition<
        fields::cellType::Yee,
        FieldB,
        T_dim
    >
    {
    };

} // namespace traits
} // namespace picongpu

namespace pmacc
{
namespace traits
{

    //! Node value traits for checkpointing
    template< >
    struct GetComponentsType<
        picongpu::fields::maxwellSolver::yeePML::NodeValues,
        false
    >
    {
        typedef picongpu::float_X type;
    };

    //! Node value traits for checkpointing
    template< >
    struct GetNComponents<
        picongpu::fields::maxwellSolver::yeePML::NodeValues,
        false
    >
    {
        static constexpr uint32_t value =
            picongpu::fields::maxwellSolver::yeePML::NodeValues::numComponents;
    };

} // namespace traits
} // namespace pmacc
