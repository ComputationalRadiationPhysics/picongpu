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
        typename T_DataBoxValue,
        typename T_DataBoxStateIdx,
        typename T_ConfigNumberDataType
        >
    class AtomicDataBox
    {
    public:

        using DataBoxValue = T_DataBoxValue;
        using DataBoxStateIdx = T_DataBoxStateIdx;
        using Idx = T_ConfigNumberDataType;
        using ValueType = typename DataBoxValue::ValueType;

    private:
        uint32_t numStates;
        DataBoxValue boxStateEnergy;
        DataBoxStateIdx boxIdxState;

        uint32_t numTransitions;
        DataBoxValue boxCollisionalOscillatorStrength;
        DataBoxValue boxCinx1;
        DataBoxValue boxCinx2;
        DataBoxValue boxCinx3;
        DataBoxValue boxCinx4;
        DataBoxValue boxCinx5;
        DataBoxStateIdx boxLowerIdx;
        DataBoxStateIdx boxUpperIdx;
        

    public:
        // Constructor
        AtomicDataBox(
            DataBoxValue boxStateEnergy,
            DataBoxStateIdx boxIdx,
            uint32_t numStates,

            DataBoxValue boxCollisionalOscillatorStrength,
            DataBoxValue boxCinx1,
            DataBoxValue boxCinx2,
            DataBoxValue boxCinx3,
            DataBoxValue boxCinx4,
            DataBoxValue boxCinx5,
            DataBoxStateIdx boxLowerIdx,
            DataBoxStateIdx boxUpperIdx,
            uint32_t numTransitions
        ):
            boxStateEnergy( boxStateEnergy ),
            boxIdx( boxIdx ),
            numStates( numStates ),

            boxLowerIdx( boxLowerIdx ),
            boxUpperIdx( boxUpperIdx ),
            boxCollisionalOscillatorStrength( boxCollisionalOscillatorStrength ),
            boxCinx1( boxCinx1 ),
            boxCinx2( boxCinx2 ),
            boxCinx3( boxCinx3 ),
            boxCinx4( boxCinx4 ),
            boxCinx5( boxCinx5 ),
            numTransitions ( numTransitions )
        {}

        // get energy, respective to ground state, of atomic state
        // @param idx ... configNumber of atomic state
        // return unit: SI_uints
        HDINLINE ValueType operator( )( Idx const idx )
        {
            // one is a special case
            if( idx == 0 )
                return 0.0_X;

            // search for state in list
            for ( uint32_t i = 0u; i < this->numStates; i++ )
            {
                if ( boxIdx( i ) == idx )
                {
                    return float_X(boxStateEnergy( i ) * picongpu::UNITCONV_eV_to_Joule) /
                        picongpu::ATOMIC_UNIT_ENERGY);
                }
            }
            // atomic state not found return zero
            return static_cast< ValueType >( 0 );
        }

        // returns index of transition in databox, numTransition qual to not found
        HDINLINE uint32_t findTransition( Idx const lowerIdx, Idx const upperIdx )
        {
            // search for transition in list
            for ( uint32_t i = 0u; i < this->numTransitions; i++ )
                if ( boxLowerIdx( i ) == lowerIdx && boxUpperIdx( i ) = upperIdx )
                    return i;
            return numTransitions;
        }

        HDINLINE Idx getUpperIdx( uint32_t transitionIndex ) const
        {
            return boxUpperIdx( transitionIndex );
        }

        HDINLINE Idx getLowerIdx( uint32_t transitionIndex ) const
        {
            return boxLowerIdx( transitionIndex );
        }

        // number of Transitions stored in this box
        HDINLINE uint32_t getNumTransitions( )
        {
            return this->numTransition;
        }

        // number of atomic states stored in this box
        HDINLINE uint32_t getNumStates( )
        {
            return this->numStates;
        }

        HDINLINE ValueType getCollisionalOscillatorStrength( uint32_t const index )
        {
            if ( index < numTransitions )
                return boxOscillatorStrength( i );

            return static_cast< ValueType >(0);
        }

        HDINLINE ValueType getCinx1( uint32_t const index )
        {
            if (index < numTransitions )
            return boxCinx1( index );
        }

        HDINLINE ValueType getCinx2( uint32_t const index )
        {
            if (index < numTransitions )
            return boxCinx2( index );
        }

        HDINLINE ValueType getCinx3( uint32_t const index )
        {
            if (index < numTransitions )
            return boxCinx3( index );
        }

        HDINLINE ValueType getCinx4( uint32_t const index )
        {
            if (index < numTransitions )
            return boxCinx4( index );
        }

        HDINLINE ValueType getCinx5( uint32_t const index )
        {
            if (index < numTransitions )
            return boxCinx5( index );
        }

        // must be called sequentially!
        HDINLINE void addLevel(
            Idx const idx,
            ValueType const energy
            )
        {
            this->boxIdx[ numStates ] = idx;
            this->boxStateEnergy [ numStates ] = energy;
            this->numStates += 1u;
        }

        // must be called sequentially!
        HDINLINE void addTransition(
            Idx const lowerIdx,
            Idx const upperIdx,
            ValueType const collisionalOscillatorStrength,
            ValueType const gauntCoefficents[5]
            )
        {
            this->boxLowerIdx[ numTransitions ] = lowerIdx;
            this->boxUpperIdx[ numTransitions ] = upperIdx;
            this->boxCollisionalOscillatorStrength[numTransitions ] = collisionalOscillatorStrength;
            this->boxCinx1 = gauntCoefficents[0];
            this->boxCinx2 = gauntCoefficents[1];
            this->boxCinx3 = gauntCoefficents[2];
            this->boxCinx4 = gauntCoefficents[3];
            this->boxCinx5 = gauntCoefficents[4];
            this->numTransitions += 1u;
        }
    };


    // Rate matrix host-device storage,
    // to be used from the host side only
    template< typename T_ConfigNumberDataType = uint64_t >
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
        using DataBoxType = AtomicDataBox<
            InternalDataBoxTypeValue,
            InternalDataBoxTypeIdx,
            T_ConfigNumberDataType
        >;

        private:
        //pointers to storage
        std::unique_ptr< BufferValue > dataStateEnergy; // unit: eV
        std::unique_ptr< BufferIdx > dataIdx;   // unit: unitless

        std::unique_ptr< BufferValue > dataOscillatorStrength; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx1; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx2; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx3; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx4; // unit: unitless
        std::unique_ptr< BufferValue > dataCinx5; // unit: unitless
        std::unique_ptr< BufferIdx > dataLowerIdx; // unit: unitless
        std::unique_ptr< BufferIdx > dataUpperIdx; // unit: unitless

        // number of states included in atomic data
        uint32_t numStates;
        uint32_t numTransitions;

    public:
        HINLINE AtomicData(
            uint32_t const numStates,
            uint32_t const numTransitions
        ):
            numStates( numStates ),
            numTransitions( numTransitions )
        {
            // get values for init of databox
            auto sizeStates = pmacc::DataSpace< 1 >::create( numStates);
            auto sizeTransitions = pmacc::DataSpace< 1 >::create( numTransitions);

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

            dataOscillatorStrength.reset(
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
        HINLINE DataBoxType getHostDataBox( )
        {
            return DataBoxTypeStates(
                dataStateEnergy->getHostBuffer( ).getDataBox( ),
                dataIdx->getHostBuffer( ).getDataBox( ),
                this->numStates,

                dataCollisionalOscillatorStrength->getHostDataBox( ).getDataBox( ),
                dataCinx1->getHostDataBox( ).getDataBox( ),
                dataCinx2->getHostDataBox( ).getDataBox( ),
                dataCinx3->getHostDataBox( ).getDataBox( ),
                dataCinx4->getHostDataBox( ).getDataBox( ),
                dataCinx5->getHostDataBox( ).getDataBox( ),
                dataLowerIdx->getHostBuffer( ).getDataBox( ),
                dataUpperIdx->getHostBuffer( ).getDataBox( ),
                this->numTransitions
                );
        }

        //! Get the device data box for the rate matrix values
        HINLINE DataBoxType getDeviceDataBox( )
        {
            return DataBoxTypeStates(
                dataStateEnergy->getDeviceBuffer( ).getDataBox( ),
                dataIdx->getDeviceBuffer( ).getDataBox( ),
                this->numStates,

                dataCollisionalOscillatorStrength->getHostDataBox( ).getDataBox( ),
                dataCinx1->getDeviceDataBox( ).getDataBox( ),
                dataCinx2->getDeviceDataBox( ).getDataBox( ),
                dataCinx3->getDeviceDataBox( ).getDataBox( ),
                dataCinx4->getDeviceDataBox( ).getDataBox( ),
                dataCinx5->getDeviceDataBox( ).getDataBox( ),
                dataLowerIdx->getDeviceBuffer( ).getDataBox( ),
                dataUpperIdx->getDeviceBuffer( ).getDataBox( ),
                this->numTransitions
                );
        }

        void syncToDevice( )
        {
            dataStateEnergy->hostToDevice( );
            dataIdx->hostToDevice( );

            dataOscillatorStrength->hostToDevice( );
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
