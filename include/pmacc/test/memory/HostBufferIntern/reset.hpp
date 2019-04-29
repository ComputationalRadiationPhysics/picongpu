/* Copyright 2015-2019 Erik Zenker
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

/* #includes in "test/memoryUT.cu" */


namespace pmacc
{
namespace test
{
namespace memory
{
namespace HostBufferIntern
{

/**
 * Checks if the HostBufferIntern is reseted correctly to zero.
 */
struct ResetTest {

    template<typename T_Dim>
    void exec(T_Dim)
    {
        using Data = uint8_t ;
        using Extents = size_t;

        using ::pmacc::test::memory::getElementsPerDim;

        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_Dim>();

        for(unsigned i = 0; i < nElementsPerDim.size(); ++i)
        {
            ::pmacc::DataSpace<T_Dim::value> const dataSpace = ::pmacc::DataSpace<T_Dim::value>::create(nElementsPerDim[i]);
            ::pmacc::HostBufferIntern<Data, T_Dim::value> hostBufferIntern(dataSpace);

            hostBufferIntern.reset();

            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i){
                BOOST_CHECK_EQUAL( hostBufferIntern.getPointer()[i], 0 );
            }

        }

    }

    PMACC_NO_NVCC_HDWARNING
    template<typename T_Dim>
    HDINLINE void operator()(T_Dim dim)
    {
        exec(dim);
    }
};

} // namespace HostBufferIntern
} // namespace memory
} // namespace test
} // namespace pmacc

BOOST_AUTO_TEST_CASE( reset )
{
    using namespace pmacc::test::memory::HostBufferIntern;
    ::boost::mpl::for_each< Dims >( ResetTest() );
}
