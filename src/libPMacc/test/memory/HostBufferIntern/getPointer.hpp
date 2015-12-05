/**
 * Copyright 2015 Erik Zenker
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

/* #includes in "test/memoryUT.cu" */

/**
 * Test if getPointer and getBasePointer provide correct results.
 */
struct GetPointerTest
{

    template<typename T_Dim>
    void operator()( T_Dim )
    {

        typedef uint8_t Data;
        typedef size_t Extents;

        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_Dim>( );
        
        for(size_t i = 0; i < nElementsPerDim.size(); ++i)
        {
            ::PMacc::DataSpace<T_Dim::value> const dataSpace = ::PMacc::DataSpace<T_Dim::value>::create( nElementsPerDim[i] );
            ::PMacc::DataSpace<T_Dim::value> const dataSpaceSmall = ::PMacc::DataSpace<T_Dim::value>::create( 1 );            
            ::PMacc::DataSpace<T_Dim::value> const offset = ::PMacc::DataSpace<T_Dim::value>::create( 1 );
            ::PMacc::DataSpace<T_Dim::value> const zeroOffset = ::PMacc::DataSpace<T_Dim::value>::create( 0 );            

            ::PMacc::HostBufferIntern<Data, T_Dim::value> hostBufferIntern( dataSpace );
            ::PMacc::HostBufferIntern<Data, T_Dim::value> hostBufferInternOffset( hostBufferIntern, dataSpaceSmall, offset );
            ::PMacc::HostBufferIntern<Data, T_Dim::value> hostBufferInternZeroOffset( hostBufferIntern, dataSpace, zeroOffset );            

            BOOST_CHECK( hostBufferIntern.getBasePointer() != NULL );
            BOOST_CHECK( hostBufferIntern.getPointer() != NULL );
            BOOST_CHECK( hostBufferInternOffset.getBasePointer() != NULL );
            BOOST_CHECK( hostBufferInternOffset.getPointer() != NULL );            
            BOOST_CHECK( hostBufferInternZeroOffset.getBasePointer() != NULL );
            BOOST_CHECK( hostBufferInternZeroOffset.getPointer() != NULL );            

            BOOST_CHECK( hostBufferIntern.getBasePointer() == hostBufferIntern.getPointer() );
            BOOST_CHECK( hostBufferInternOffset.getBasePointer() != hostBufferInternOffset.getPointer() );
            BOOST_CHECK( hostBufferInternZeroOffset.getBasePointer() == hostBufferInternZeroOffset.getPointer() );

            BOOST_CHECK( hostBufferInternZeroOffset.getBasePointer() == hostBufferIntern.getBasePointer() );
            BOOST_CHECK( hostBufferInternZeroOffset.getPointer() == hostBufferIntern.getPointer() );            
            
        }

    }

};

BOOST_AUTO_TEST_CASE( getPointer ){
    ::boost::mpl::for_each< Dims >( GetPointerTest() );

}
