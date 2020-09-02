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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdint>
#include <memory>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{

    // Data box type for rate matrix on host and device
    template< typename T_DataBox >
    class RateMatrixBox
    {
    public:

        using DataBox = T_DataBox;
        using Idx = uint32_t; // also probably becomes a template parameter
        using ValueType = typename DataBox::ValueType;


        RateMatrixBox(
            DataBox box,
            Idx firstStateIndex
        ):
            box( box ),
            firstStateIndex( firstStateIndex )
        {
        }


        HDINLINE ValueType & operator( )( Idx const idx )
        {
            return box( idx - firstStateIndex );
        }

        HDINLINE ValueType operator( )( Idx const idx ) const
        {
            // one is a special case
            if( idx == 1 )
                return 0.0_X;
            return box( idx - firstStateIndex );
        }

    private:

        DataBox box;
        Idx firstStateIndex;

    };


    // Rate matrix host-device storage,
    // to be used from the host side only
    class RateMatrix
    {
    public:

        // underlying int index type used for states,
        // will probably become a template parameter of this class later
        using Idx = uint32_t;

        using Buffer = pmacc::GridBuffer<
            float_X,
            1
        >;

        using InternalDataBoxType = pmacc::DataBox<
            pmacc::PitchedBox<
                float_X,
                1
            >
        >;

        using DataBoxType = RateMatrixBox< InternalDataBoxType >;

        HINLINE RateMatrix(
            Idx const firstStateIndex,
            Idx const numStates
        ):
            firstStateIndex( firstStateIndex )
        {
            auto size = pmacc::DataSpace< 1 >::create( numStates );
            auto const guardSize = pmacc::DataSpace< 1 >::create( 0 );
            auto const layout = pmacc::GridLayout< 1 >(
                size,
                guardSize
            );
            data.reset(
                new Buffer( layout )
            );
        }

        //! Get the host data box for the rate matrix values
        HINLINE DataBoxType getHostDataBox( )
        {
            return DataBoxType(
                data->getHostBuffer( ).getDataBox( ),
                firstStateIndex
            );
        }

        //! Get the device data box for the rate matrix values
        HINLINE DataBoxType getDeviceDataBox( )
        {
            return DataBoxType(
                data->getDeviceBuffer( ).getDataBox( ),
                firstStateIndex
            );
        }

        void syncToDevice( )
        {
            data->hostToDevice( );
        }

    private:

        std::unique_ptr< Buffer > data;
        Idx firstStateIndex;

    };

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
