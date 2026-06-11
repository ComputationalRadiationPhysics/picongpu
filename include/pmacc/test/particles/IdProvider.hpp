/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/IdProvider.hpp>
#include <pmacc/types.hpp>

#include <algorithm>
#include <cstdint>
#include <set>

#include <catch2/catch_test_macros.hpp>

namespace pmacc
{
    namespace test
    {
        namespace particles
        {
            template<uint32_t T_numIdsPerBlock>
            struct GenerateIds
            {
                template<class T_Box, typename T_IdGenerator, typename T_Worker>
                HDINLINE void operator()(
                    T_Worker const& worker,
                    T_Box outputbox,
                    T_IdGenerator idGenerator,
                    uint32_t numThreads,
                    uint32_t numIdsPerThread) const
                {
                    using namespace ::pmacc;

                    uint32_t const blockId = worker.blockDomIdxND().x() * T_numIdsPerBlock;

                    lockstep::makeForEach<T_numIdsPerBlock>(worker)(
                        [&](uint32_t const linearId)
                        {
                            uint32_t const localId = blockId + linearId;
                            if(localId < numThreads)
                            {
                                for(uint32_t i = 0u; i < numIdsPerThread; i++)
                                {
                                    uint32_t x = idGenerator.fetchInc(worker);
                                    outputbox(i * numThreads + localId) = x;
                                }
                            }
                        });
                }
            };

            /** function checks if a value is in a collection
             *
             * Use like: REQUIRE(checkDuplicate(col, value, true|false));
             * @param col Container to be searched
             * @param value Value to search for
             * @param shouldFind Whether the value is expected in the collection or not
             * @return Error-Value, if the value is not found and shouldFind is true or
             *         the value is found and shouldFind is false, otherwise a True-Value
             */
            template<class T_Collection, typename T>
            bool checkDuplicate(T_Collection const& col, T const& value, bool shouldFind)
            {
                if((std::find(col.begin(), col.end(), value) != col.end()) != shouldFind)
                {
                    bool res(false);
                    if(shouldFind)
                        std::cout << "Value not found found: ";
                    else
                        std::cout << "Duplicate found: ";
                    std::cout << value << ". Values=[";
                    for(typename T_Collection::const_iterator it = col.begin(); it != col.end(); ++it)
                        std::cout << *it << ",";
                    std::cout << "]";
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

                    uint64_t maxRanks = Environment<T_dim>::get().GridController().getGpuNodes().productOfComponents();
                    uint64_t rank = Environment<T_dim>::get().GridController().getScalarPosition();
                    auto idProvider = IdProvider("id provider", rank, maxRanks);

                    // Check initial state
                    auto state = idProvider.getState();
                    REQUIRE(state.startId == state.nextId);
                    REQUIRE(state.maxNumProc == 1u);
                    REQUIRE(!idProvider.isOverflown());
                    std::set<uint64_t> ids;
                    REQUIRE(idProvider.getNewIdHost() == state.nextId);
                    // Generate some IDs using the function
                    for(int i = 0; i < numIds; i++)
                    {
                        uint64_t const newId = idProvider.getNewIdHost();
                        REQUIRE(checkDuplicate(ids, newId, false));
                        ids.insert(newId);
                    }
                    // Reset the state
                    idProvider.setState(state);
                    REQUIRE(idProvider.getNewIdHost() == state.nextId);
                    // Generate the same IDs on the device
                    HostDeviceBuffer<uint64_t, 1> idBuf(numIds);

                    PMACC_LOCKSTEP_KERNEL(GenerateIds<numIdsPerBlock>{})
                        .template config<numIdsPerBlock>(numBlocks)(
                            idBuf.getDeviceBuffer().getDataBox(),
                            idProvider.getDeviceGenerator(),
                            numThreads,
                            numIdsPerThread);
                    idBuf.deviceToHost();
                    REQUIRE(numIds == ids.size());
                    auto hostBox = idBuf.getHostBuffer().getDataBox();
                    // Make sure they are the same
                    for(uint32_t i = 0; i < numIds; i++)
                    {
                        REQUIRE(checkDuplicate(ids, hostBox(i), true));
                    }
                }
            };

        } // namespace particles
    } // namespace test
} // namespace pmacc

TEST_CASE("particles::IDProvider", "[IDProvider]")
{
    using namespace pmacc::test::particles;
    IdProviderTest<TEST_DIM>()();
}
