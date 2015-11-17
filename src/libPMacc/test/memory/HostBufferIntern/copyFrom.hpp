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

// #includes in "test/memoryUT.cu"

/**
 * Checks if data is copied correctly from device to
 * host.
 */
struct CopyFromTest {

    template<typename T_DIM>
    void operator()(T_DIM){

        typedef uint8_t Data;
        typedef size_t  Extents;

        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_DIM>();
        
        for(unsigned i = 0; i < nElementsPerDim.size(); ++i){
            ::PMacc::DataSpace<T_DIM::value> const dataSpace              = ::PMacc::DataSpace<T_DIM::value>::create(nElementsPerDim[i]);
            ::PMacc::HostBuffer<Data, T_DIM::value>* hostBufferIntern     = new ::PMacc::HostBufferIntern<Data, T_DIM::value>(dataSpace);
            ::PMacc::DeviceBuffer<Data, T_DIM::value>* deviceBufferIntern = new ::PMacc::DeviceBufferIntern<Data, T_DIM::value>(dataSpace);

            hostBufferIntern->reset();

            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i){
                hostBufferIntern->getPointer()[i] = static_cast<Data>(i);
            }

            deviceBufferIntern->copyFrom(*hostBufferIntern);
            hostBufferIntern->reset();
            hostBufferIntern->copyFrom(*deviceBufferIntern);

            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i){
                BOOST_CHECK_EQUAL(hostBufferIntern->getPointer()[i], static_cast<Data>(i));
            }

            delete hostBufferIntern;
            delete deviceBufferIntern;

        }

    }

};

BOOST_AUTO_TEST_CASE( copyFrom ){
    ::boost::mpl::for_each< Dims >( CopyFromTest() );

}
