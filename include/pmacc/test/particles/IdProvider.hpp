/* Copyright 2016-2019 Alexander Grund
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

#include <pmacc/types.hpp>
#include <pmacc/particles/IdProvider.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/eventSystem/EventSystem.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/test/unit_test.hpp>
#include <set>
#include <algorithm>
#include <stdint.h>

BOOST_AUTO_TEST_SUITE( particles )


namespace pmacc
{
namespace test
{
namespace particles
{
    namespace bmpl = boost::mpl;

    template<
        uint32_t T_numWorkers,
        uint32_t T_numIdsPerBlock,
        typename T_IdProvider
    >
    struct GenerateIds
    {
        template<class T_Box, typename T_Acc>
        HDINLINE void operator()(const T_Acc & acc, T_Box outputbox, uint32_t numThreads, uint32_t numIdsPerThread) const
        {
            using namespace ::pmacc;
            using namespace mappings::threads;

            constexpr uint32_t numWorkers = T_numWorkers;

            uint32_t const workerIdx = threadIdx.x;

            uint32_t const blockId = blockIdx.x * T_numIdsPerBlock;
            ForEachIdx<
                IdxConfig<
                    T_numIdsPerBlock,
                    numWorkers
                >
            >{ workerIdx }(
                [&](
                    uint32_t const linearId,
                    uint32_t const
                )
                {
                    uint32_t const localId = blockId + linearId;
                    if( localId < numThreads )
                    {
                        for( uint32_t i = 0u; i < numIdsPerThread; i++ )
                            outputbox( i * numThreads + localId ) = T_IdProvider::getNewId( );
                    }
                }
            );


        }
    };

/**
 * Boost.Test compatible function that checks if a value is in a collection
 * Use like: BOOST_REQUIRE(checkDuplicate(col, value, true|false));
 * @param col Container to be searched
 * @param value Value to search for
 * @param shouldFind Whether the value is expected in the collection or not
 * @return Error-Value, if the value is not found and shouldFind is true or
 *         the value is found and shouldFind is false, otherwise a True-Value
 */
template<class T_Collection, typename T>
boost::test_tools::predicate_result
checkDuplicate(const T_Collection& col, const T& value, bool shouldFind)
{
    if((std::find(col.begin(), col.end(), value) != col.end()) != shouldFind)
    {
        boost::test_tools::predicate_result res(false);
        if(shouldFind)
            res.message() << "Value not found found: ";
        else
            res.message() << "Duplicate found: ";
        res.message() << value << ". Values=[";
        for(typename T_Collection::const_iterator it = col.begin(); it != col.end(); ++it)
            res.message() << *it << ",";
        res.message() << "]";
        return res;
    }

    return true;
}


template<unsigned T_dim>
struct IdProviderTest
{
    void operator()()
    {
        using namespace ::pmacc;

        constexpr uint32_t numBlocks = 4;
        constexpr uint32_t numIdsPerBlock = 64;
        constexpr uint32_t numThreads = numBlocks * numIdsPerBlock;
        constexpr uint32_t numIdsPerThread = 2;
        constexpr uint32_t numIds = numThreads * numIdsPerThread;

        using IdProvider = IdProvider< T_dim >;
        IdProvider::init();
        // Check initial state
        typename IdProvider::State state = IdProvider::getState();
        BOOST_REQUIRE_EQUAL(state.startId, state.nextId);
        BOOST_REQUIRE_EQUAL(state.maxNumProc, 1u);
        BOOST_REQUIRE(!IdProvider::isOverflown());
        std::set<uint64_t> ids;
        BOOST_REQUIRE_EQUAL(IdProvider::getNewIdHost(), state.nextId);
        // Generate some IDs using the function
        for(int i=0; i<numIds; i++)
        {
            const uint64_t newId = IdProvider::getNewIdHost();
            BOOST_REQUIRE(checkDuplicate(ids, newId, false));
            ids.insert(newId);
        }
        // Reset the state
        IdProvider::setState(state);
        BOOST_REQUIRE_EQUAL(IdProvider::getNewIdHost(), state.nextId);
        // Generate the same IDs on the device
        HostDeviceBuffer< uint64_t, 1 > idBuf(numIds);
        constexpr uint32_t numWorkers = traits::GetNumWorkers<
            numIdsPerBlock
        >::value;
        PMACC_KERNEL( GenerateIds<
            numWorkers,
            numIdsPerBlock,
            IdProvider
        >{  })(
            numBlocks,
            numWorkers
        )(
            idBuf.getDeviceBuffer().getDataBox(),
            numThreads,
            numIdsPerThread
        );
        idBuf.deviceToHost();
        BOOST_REQUIRE_EQUAL(numIds, ids.size());
        auto hostBox = idBuf.getHostBuffer().getDataBox();
        // Make sure they are the same
        for(uint32_t i=0; i<numIds; i++)
        {
            BOOST_REQUIRE(checkDuplicate(ids, hostBox(i), true));
        }
    }
};

} // namespace particles
} // namespace test
} // namespace pmacc

BOOST_AUTO_TEST_CASE(IdProvider)
{
    using namespace pmacc::test::particles;
    IdProviderTest<TEST_DIM>()();
}

BOOST_AUTO_TEST_SUITE_END()
