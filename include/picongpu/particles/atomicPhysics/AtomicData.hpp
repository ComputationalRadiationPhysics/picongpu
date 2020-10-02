/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include "picongpu/param/physicalConstants.param"

#include <cstdint>
#include <memory>
#include <utility>

#pragma once


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
    /** too different classes giving acess to atomic data:
     * - base class ... implements actual functionality
     * dataBox class ... provides acess implementation for actual storage in box
     *      encapsulates index shift currently
     */

    // Data box type for rate matrix on host and device
    template<
        uint8_t T_atomicNumber,
        typename T_DataBoxValue,
        typename T_DataBoxStateIdx,
        typename T_ConfigNumberDataType,
        uint32_t T_maxNumberStates,
        uint32_t T_maxNumberTransitions
        >
    class AtomicDataBox
    {
    public:

        using DataBoxValue = T_DataBoxValue;
        using DataBoxStateIdx = T_DataBoxStateIdx;
        using Idx = T_ConfigNumberDataType;
        using ValueType = typename DataBoxValue::ValueType;

    private:
        DataBoxValue m_boxStateEnergy;
        DataBoxStateIdx m_boxStateIdx;
        uint32_t m_numStates;

        uint32_t m_numTransitions;
        DataBoxValue m_boxCollisionalOscillatorStrength;
        DataBoxValue m_boxCinx1;
        DataBoxValue m_boxCinx2;
        DataBoxValue m_boxCinx3;
        DataBoxValue m_boxCinx4;
        DataBoxValue m_boxCinx5;
        DataBoxStateIdx m_boxLowerIdx;
        DataBoxStateIdx m_boxUpperIdx;
        

    public:
        // Constructor
        AtomicDataBox(
            uint32_t numStates,
            DataBoxValue boxStateEnergy,
            DataBoxStateIdx boxStateIdx,

            DataBoxStateIdx boxLowerIdx,
            DataBoxStateIdx boxUpperIdx,
            DataBoxValue boxCollisionalOscillatorStrength,
            DataBoxValue boxCinx1,
            DataBoxValue boxCinx2,
            DataBoxValue boxCinx3,
            DataBoxValue boxCinx4,
            DataBoxValue boxCinx5,
            uint32_t numTransition
        ):
            m_boxStateEnergy( boxStateEnergy ),
            m_boxStateIdx( boxStateIdx ),
            m_boxLowerIdx( boxLowerIdx ),
            m_boxUpperIdx( boxUpperIdx ),
            m_boxCollisionalOscillatorStrength( boxCollisionalOscillatorStrength ),
            m_boxCinx1( boxCinx1 ),
            m_boxCinx2( boxCinx2 ),
            m_boxCinx3( boxCinx3 ),
            m_boxCinx4( boxCinx4 ),
            m_boxCinx5( boxCinx5 ),
            m_numStates( numStates ),
            m_numTransitions( numTransitions )
        {

        }

        // get energy, respective to ground state, of atomic state
        // @param idx ... configNumber of atomic state
        // return unit: SI_uints
        HDINLINE ValueType operator( )( Idx const idx )
        {
            // one is a special case
            if( idx == 0 )
                return 0.0_X;

            // search for state in list
            for ( uint32_t i = 0u; i < this->m_numStates; i++ )
            {
                if ( m_boxStateIdx( i ) == idx )
                {
                    return static_cast< float_X >( (m_boxStateEnergy( i ) * picongpu::UNITCONV_eV_to_Joule) /
                        picongpu::SI::ATOMIC_UNIT_ENERGY );
                }
            }
            // atomic state not found return zero
            return static_cast< ValueType >( 0 );
        }

        HDINLINE Idx getAtomicStateConfigNumberIndex( uint32_t indexState )
        {
            return this->m_boxStateIdx( indexState );
        }

        // returns index of transition in databox, numTransition qual to not found
        HDINLINE uint32_t findTransition( Idx const lowerIdx, Idx const upperIdx )
        {
            // search for transition in list
            for ( uint32_t i = 0u; i < this->m_numTransitions; i++ )
                if ( this->m_boxLowerIdx( i ) == lowerIdx && this->m_boxUpperIdx( i ) == upperIdx )
                    return i;
            return this->m_numTransitions;
        }

        HDINLINE Idx getUpperIdxTransition( uint32_t transitionIndex ) const
        {
            return this->m_boxUpperIdx( transitionIndex );
        }

        HDINLINE Idx getLowerIdxTransition( uint32_t transitionIndex ) const
        {
            return this->m_boxLowerIdx( transitionIndex );
        }

        // number of Transitions stored in this box
        HDINLINE uint32_t getNumTransitions( ) const
        {
            return this->m_numTransitions;
        }

        // number of atomic states stored in this box
        HDINLINE uint32_t getNumStates( ) const
        {
            return this->m_numStates;
        }

        HDINLINE ValueType getCollisionalOscillatorStrength( uint32_t const indexTransition ) const
        {
            if ( indexTransition < this->m_numTransitions )
                return this->m_boxCollisionalOscillatorStrength( indexTransition );

            return static_cast< ValueType >(0);
        }

        HDINLINE ValueType getCinx1( uint32_t const indexTransition ) const
        {
            if ( indexTransition < this->m_numTransitions )
                return this->m_boxCinx1( indexTransition );
            return static_cast< ValueType >(0);
        }

        HDINLINE ValueType getCinx2( uint32_t const indexTransition ) const
        {
            if ( indexTransition < this->m_numTransitions )
                return this->m_boxCinx2( indexTransition );
            return static_cast< ValueType >(0);
        }

        HDINLINE ValueType getCinx3( uint32_t const indexTransition ) const
        {
            if ( indexTransition < this->m_numTransitions )
                return this->m_boxCinx3( indexTransition );
            return static_cast< ValueType >(0);
        }

        HDINLINE ValueType getCinx4( uint32_t const indexTransition ) const
        {
            if (indexTransition < this->m_numTransitions )
                return this->m_boxCinx4( indexTransition );
            return static_cast< ValueType >(0);
        }

        HDINLINE ValueType getCinx5( uint32_t const indexTransition ) const
        {
            if ( indexTransition < this->m_numTransitions )
                return this->m_boxCinx5( indexTransition );
            return static_cast< ValueType >(0);
        }

        HDINLINE constexpr static uint8_t getAtomicNumber()
        {
            return T_atomicNumber;
        }

        // must be called sequentially!
        // assumes no more than levels are added than memory is available
        HDINLINE void addLevel(
            Idx const idx, // must be index as defined in ConfigNumber
            ValueType const energy // unit: eV
            )
        {
            if ( this->m_numStates < T_maxNumberStates )
            {
                this->m_boxStateIdx[ this->m_numStates ] = idx;
                this->m_boxStateEnergy [ this->m_numStates ] = energy;
                this->m_numStates += 1u;
            }
        }

        // must be called sequentially!
        HDINLINE void addTransition(
            Idx const lowerIdx, // must be index as defined in ConfigNumber
            Idx const upperIdx, // must be index as defined in ConfigNumber
            ValueType const collisionalOscillatorStrength,
            ValueType const gauntCoefficent1,
            ValueType const gauntCoefficent2,
            ValueType const gauntCoefficent3,
            ValueType const gauntCoefficent4,
            ValueType const gauntCoefficent5
            )
        {
            if( this->m_numTransitions <= T_maxNumberTransitions )
            {
                this->m_boxLowerIdx[ m_numTransitions ] = lowerIdx;
                this->m_boxUpperIdx[ m_numTransitions ] = upperIdx;
                this->m_boxCollisionalOscillatorStrength[ m_numTransitions ] = collisionalOscillatorStrength;
                this->m_boxCinx1[ m_numTransitions ] = gauntCoefficent1;
                this->m_boxCinx2[ m_numTransitions ] = gauntCoefficent2;
                this->m_boxCinx3[ m_numTransitions ] = gauntCoefficent3;
                this->m_boxCinx4[ m_numTransitions ] = gauntCoefficent4;
                this->m_boxCinx5[ m_numTransitions ] = gauntCoefficent5;
                this->m_numTransitions += 1u;
            }
        }
    };


    // Rate matrix host-device storage,
    // to be used from the host side only
    template<
        uint8_t T_atomicNumber,
        typename T_ConfigNumberDataType = uint64_t
        >
    class AtomicData
    {
    // type declarations
    public:
        // underlying int index type used for states,
        // will probably become a template parameter of this class later
        using Idx = T_ConfigNumberDataType;
        using BufferValue = pmacc::GridBuffer<
            float_X,
            1
        >;
        using BufferIdx = pmacc::GridBuffer<
            T_ConfigNumberDataType,
            1
        >;
        // data storage
        using InternalDataBoxTypeValue = pmacc::DataBox<
            pmacc::PitchedBox<
                float_X,
                1
            >
        >;
        using InternalDataBoxTypeIdx = pmacc::DataBox<
            pmacc::PitchedBox<
                T_ConfigNumberDataType,
                1
            >
        >;

        // acess datatype used on device
        template<
            uint32_t T_maxNumberStates,
            uint32_t T_maxNumberTransitions
        >
        using DataBoxType = AtomicDataBox<
            T_atomicNumber,
            InternalDataBoxTypeValue,
            InternalDataBoxTypeIdx,
            T_ConfigNumberDataType,
            T_maxNumberStates,
            T_maxNumberTransitions
        >;

    private:
        //pointers to storage
        std::unique_ptr< BufferValue > dataStateEnergy; // unit: eV
        std::unique_ptr< BufferIdx > dataIdx;   // unit: unitless

        std::unique_ptr< BufferValue > dataCollisionalOscillatorStrength; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx1; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx2; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx3; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx4; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx5; // unit: unitless
        std::unique_ptr< BufferIdx > dataLowerIdx; // unit: unitless
        std::unique_ptr< BufferIdx > dataUpperIdx; // unit: unitless

        // number of states included in atomic data
        uint32_t m_maxNumberStates;
        uint32_t m_maxNumberTransitions;


    public:
        HINLINE AtomicData(
            uint32_t maxNumberStates,
            uint32_t maxNumberTransitions
            )
        {
            m_maxNumberStates = maxNumberStates;
            m_maxNumberTransitions = maxNumberTransitions;

            // get values for init of databox
            auto sizeStates = pmacc::DataSpace< 1 >::create( m_maxNumberStates);
            auto sizeTransitions = pmacc::DataSpace< 1 >::create( m_maxNumberTransitions);

            auto const guardSize = pmacc::DataSpace< 1 >::create( 0 );

            auto const layoutStates = pmacc::GridLayout< 1 >(
                sizeStates,
                guardSize
            );
            auto const layoutTransitions = pmacc::GridLayout< 1 >(
                sizeTransitions,
                guardSize
            );

            // create Buffers on stack and store pointer to it as member
            dataStateEnergy.reset(
                new BufferValue( layoutStates )
            );
            dataIdx.reset(
                new BufferIdx( layoutStates )
            );

            dataCollisionalOscillatorStrength.reset(
                new BufferValue( layoutTransitions )
            );
            dataCinx1.reset(
                new BufferValue( layoutTransitions )
            );
            dataCinx2.reset(
                new BufferValue( layoutTransitions )
            );
            dataCinx3.reset(
                new BufferValue( layoutTransitions )
            );
            dataCinx4.reset(
                new BufferValue( layoutTransitions )
            );
            dataCinx5.reset(
                new BufferValue( layoutTransitions )
            );
            dataLowerIdx.reset(
                new BufferIdx( layoutTransitions )
            );
            dataUpperIdx.reset(
                new BufferIdx( layoutTransitions )
            );
        }

        //! Get the host data box for the rate matrix values
        template<
            uint32_t T_maxNumberStates,
            uint32_t T_maxNumberTransitions
        >
        HINLINE DataBoxType< T_maxNumberStates, T_maxNumberTransitions > getHostDataBox( )
        {
            return DataBoxType<
                T_maxNumberStates,
                T_maxNumberTransitions
            >(
                dataStateEnergy->getHostBuffer( ).getDataBox( ),
                dataIdx->getHostBuffer( ).getDataBox( ),
                0, // numStates, always fill on hostside

                dataLowerIdx->getHostBuffer( ).getDataBox( ),
                dataUpperIdx->getHostBuffer( ).getDataBox( ),
                dataCollisionalOscillatorStrength->getHostBuffer( ).getDataBox( ),
                dataCinx1->getHostBuffer( ).getDataBox( ),
                dataCinx2->getHostBuffer( ).getDataBox( ),
                dataCinx3->getHostBuffer( ).getDataBox( ),
                dataCinx4->getHostBuffer( ).getDataBox( ),
                dataCinx5->getHostBuffer( ).getDataBox( ),
                0   // numTransitions, always fill on hostside
                );
        }

        //! Get the device data box for the rate matrix values
        template<
            uint32_t T_maxNumberStates,
            uint32_t T_maxNumberTransitions
        >
        HINLINE DataBoxType< T_maxNumberStates, T_maxNumberTransitions > getDeviceDataBox( )
        {
            return DataBoxType<
                T_maxNumberStates,
                T_maxNumberTransitions
            >(
                dataStateEnergy->getDeviceBuffer( ).getDataBox( ),
                dataIdx->getDeviceBuffer( ).getDataBox( ),
                T_maxNumberStates,

                dataLowerIdx->getDeviceBuffer( ).getDataBox( ),
                dataUpperIdx->getDeviceBuffer( ).getDataBox( ),
                dataCollisionalOscillatorStrength->getHostBuffer( ).getDataBox( ),
                dataCinx1->getDeviceBuffer( ).getDataBox( ),
                dataCinx2->getDeviceBuffer( ).getDataBox( ),
                dataCinx3->getDeviceBuffer( ).getDataBox( ),
                dataCinx4->getDeviceBuffer( ).getDataBox( ),
                dataCinx5->getDeviceBuffer( ).getDataBox( ),
                T_maxNumberTransitions
                );
        }

        void syncToDevice( )
        {
            dataStateEnergy->hostToDevice( );
            dataIdx->hostToDevice( );

            dataCollisionalOscillatorStrength->hostToDevice( );
            dataCinx1->hostToDevice( );
            dataCinx2->hostToDevice( );
            dataCinx3->hostToDevice( );
            dataCinx4->hostToDevice( );
            dataCinx5->hostToDevice( );
            dataLowerIdx->hostToDevice( );
            dataUpperIdx->hostToDevice( );
        }
    };

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
