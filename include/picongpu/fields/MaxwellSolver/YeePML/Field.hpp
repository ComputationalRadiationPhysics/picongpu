/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
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
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"
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
#include <stdexcept>
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
         * version.Throws for indices out of range.
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
         * version.Throws for indices out of range.
         *
         * @param idx index less than 6
         */
        float_X const & operator[ ]( uint32_t const idx ) const
        {
            switch( idx )
            {
                case 0: return xy;
                case 1: return xz;
                case 2: return yx;
                case 3: return yz;
                case 4: return zx;
                case 5: return zy;
            }
            throw std::out_of_range(
                "In NodeValues::operator() the index = " +
                std::to_string( idx ) + " is invalid"
            );
        }

    };

    //! Base class for field in PML
    class Field : public SimulationFieldHelper< MappingDesc >, public ISimulationData
    {
    public:

        using ValueType = NodeValues;
        static constexpr int numComponents = NodeValues::numComponents;
        using UnitValueType = pmacc::math::Vector< float_64, numComponents >;

        typedef DataBox< PitchedBox< ValueType, simDim > > DataBoxType;

        typedef MappingDesc::SuperCellSize SuperCellSize;

        Field( MappingDesc cellDescription);

        virtual void reset( uint32_t currentStep );

        virtual EventTask asyncCommunication( EventTask serialEvent );

        DataBoxType getHostDataBox( );

        GridLayout< simDim > getGridLayout( );

        DataBoxType getDeviceDataBox( );

        GridBuffer< ValueType, simDim > & getGridBuffer( );

        void synchronize( );

        void syncToDevice( );

    private:

        using Buffer = pmacc::GridBuffer<
            ValueType,
            simDim
        >;
        std::unique_ptr< Buffer > data;
    };

    //! Additional electric field components in PML
    class FieldE : public Field
    {
    public:

        FieldE( MappingDesc cellDescription):
            Field( cellDescription )
        {
        }

        SimulationDataId getUniqueId( )
        {
            return getName( );
        }

        HDINLINE static UnitValueType getUnit( )
        {
            return UnitValueType::create( UNIT_EFIELD );
        }

        /** powers of the 7 base measures
         *
         * characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         * thermodynamic temperature theta, amount of substance N,
         * luminous intensity J) */
        HINLINE static std::vector< float_64 > getUnitDimension( )
        {
            return picongpu::FieldE::getUnitDimension( );
        }

        static std::string getName( )
        {
            return "PML E components";
        }

    };

    //! Additional magnetic field components in PML
    class FieldB : public Field
    {
    public:

        FieldB( MappingDesc cellDescription):
            Field( cellDescription )
        {
        }

        SimulationDataId getUniqueId( )
        {
            return getName( );
        }

        HDINLINE static UnitValueType getUnit( )
        {
            return UnitValueType::create( UNIT_BFIELD );
        }

        /** powers of the 7 base measures
         *
         * characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         * thermodynamic temperature theta, amount of substance N,
         * luminous intensity J) */
        HINLINE static std::vector< float_64 > getUnitDimension( )
        {
            return picongpu::FieldB::getUnitDimension( );
        }

        static std::string getName( )
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
        numericalCellTypes::YeeCell,
        fields::maxwellSolver::yeePML::FieldE,
        T_dim
    > : FieldPosition<
        numericalCellTypes::YeeCell,
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
        numericalCellTypes::YeeCell,
        fields::maxwellSolver::yeePML::FieldB,
        T_dim
    > : FieldPosition<
        numericalCellTypes::YeeCell,
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
