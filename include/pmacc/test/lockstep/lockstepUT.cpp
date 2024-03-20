/* Copyright 2015-2023 Erik Zenker, Alexander Grund
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

#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>
#include <pmacc/verify.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

/** @file
 *
 *  This file is testing common lockstep pattern.
 *  There are many code duplications, those are necessary because the code snippets are include into the documentation.
 */

using MyPMaccFixture = pmacc::test::PMaccFixture<TEST_DIM>;
static MyPMaccFixture fixture;

constexpr uint32_t numElements = 4096u;

// doc-include-start: lockstep generic kernel
struct IotaGenericKernel
{
    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        constexpr uint32_t blockDomSize = T_Worker::blockDomSize();
        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        // grid-strided loop over the chunked data
        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = idx;
                    }
                });
        }
    }
};

template<uint32_t T_chunkSize, typename T_DeviceBuffer>
inline void iotaGerneric(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.getCurrentSize();
    // use only half of the blocks needed to process the full data
    uint32_t const numBlocks = bufferSize / T_chunkSize / 2u;
    PMACC_LOCKSTEP_KERNEL(IotaGenericKernel{}).config<T_chunkSize>(numBlocks)(devBuffer.getDataBox(), bufferSize);
}
// doc-include-end: lockstep generic kernel

// doc-include-start: lockstep generic kernel buffer selected domain size
namespace pmacc::lockstep::traits
{
    //! Specialization to create a lockstep block configuration out of a device buffer.
    template<>
    struct MakeBlockCfg<pmacc::DeviceBuffer<uint32_t, DIM1>> : std::true_type
    {
        using type = BlockCfg<math::CT::UInt32<53>>;
    };
} // namespace pmacc::lockstep::traits

template<typename T_DeviceBuffer>
inline void iotaGernericBufferDerivedChunksize(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.getCurrentSize();
    constexpr uint32_t numBlocks = 9;
    PMACC_LOCKSTEP_KERNEL(IotaGenericKernel{}).config(numBlocks, devBuffer)(devBuffer.getDataBox(), bufferSize);
}
// doc-include-end: lockstep generic kernel buffer selected domain size

// doc-include-start: lockstep generic kernel hard coded domain size
struct IotaFixedChunkSizeKernel
{
    static constexpr uint32_t blockDomSize = 42;

    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        static_assert(blockDomSize == T_Worker::blockDomSize());

        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        // grid-strided loop over the chunked data
        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = idx;
                    }
                });
        }
    }
};

template<typename T_DeviceBuffer>
inline void iotaFixedChunkSize(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.getCurrentSize();
    constexpr uint32_t numBlocks = 10;
    PMACC_LOCKSTEP_KERNEL(IotaFixedChunkSizeKernel{}).config(numBlocks)(devBuffer.getDataBox(), bufferSize);
}
// doc-include-end: lockstep generic kernel hard coded domain size

// doc-include-start: lockstep generic kernel hard coded N dimensional domain size
struct IotaFixedChunkSizeKernelND
{
    using BlockDomSizeND = pmacc::math::CT::UInt32<42>;

    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        static constexpr uint32_t blockDomSize = BlockDomSizeND::x::value;

        static_assert(blockDomSize == T_Worker::blockDomSize());

        // grid-strided loop over the chunked data
        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = idx;
                    }
                });
        }
    }
};

template<typename T_DeviceBuffer>
inline void iotaFixedChunkSizeND(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.getCurrentSize();
    constexpr uint32_t numBlocks = 11;
    PMACC_LOCKSTEP_KERNEL(IotaFixedChunkSizeKernelND{}).config(numBlocks)(devBuffer.getDataBox(), bufferSize);
}
// doc-include-end: lockstep generic kernel hard coded N dimensional domain size

// doc-include-start: lockstep generic kernel with dynamic shared memory
struct IotaGenericKernelWithDynSharedMem
{
    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        constexpr uint32_t blockDomSize = T_Worker::blockDomSize();
        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        uint32_t* s_mem = ::alpaka::getDynSharedMem<uint32_t>(worker.getAcc());

        // grid-strided loop over the chunked data
        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    s_mem[inBlockIdx] = idx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = s_mem[inBlockIdx];
                    }
                });
        }
    }
};

template<uint32_t T_chunkSize, typename T_DeviceBuffer>
inline void iotaGernericWithDynSharedMem(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.getCurrentSize();
    // use only half of the blocks needed to process the full data
    uint32_t const numBlocks = bufferSize / T_chunkSize / 2u;
    constexpr size_t requiredSharedMemBytes = T_chunkSize * sizeof(uint32_t);
    PMACC_LOCKSTEP_KERNEL(IotaGenericKernelWithDynSharedMem{})
        .configSMem<T_chunkSize>(numBlocks, requiredSharedMemBytes)(devBuffer.getDataBox(), bufferSize);
}
// doc-include-end: lockstep generic kernel with dynamic shared memory

template<typename T_HostBuffer>
void validate(T_HostBuffer& results, T_HostBuffer& reference)
{
    auto refBufferSize = reference.getCurrentSize();
    auto* refPtr = reference.data();

    auto resultBufferSize = results.getCurrentSize();
    auto* resultPtr = results.data();

    PMACC_VERIFY(resultBufferSize == refBufferSize);
    for(uint32_t i = 0u; i < refBufferSize; ++i)
    {
        REQUIRE(refPtr[i] == resultPtr[i]);
    }
}

TEST_CASE("lockstep kernel", "[iota]")
{
    using namespace pmacc;

    auto referenceBuffer = HostBuffer<uint32_t, DIM1>(DataSpace<DIM1>{numElements});

    auto* refHostPtr = referenceBuffer.data();
    for(uint32_t i = 0u; i < numElements; ++i)
    {
        refHostPtr[i] = i;
    }

    auto hostDeviceBuffer = HostDeviceBuffer<uint32_t, DIM1>(DataSpace<DIM1>{numElements});
    using DeviceBuf = DeviceBuffer<uint32_t, DIM1>;

    // register all required test functions
    auto testsFunctions = std::make_tuple(
        // generic host size chunk size selection
        iotaGerneric<128, DeviceBuf>,
        iotaGerneric<16, DeviceBuf>,
        // generic host size chunk size selection and dynamic shared memory
        iotaGernericWithDynSharedMem<23, DeviceBuf>,
        // derive the chunk size from the result buffer
        iotaGernericBufferDerivedChunksize<DeviceBuf>,
        // kernel defined fixed chunk size (kernel defines value blockDomSize)
        iotaFixedChunkSize<DeviceBuf>,
        // kernel defined fixed chunk size (kernel defines type BlockDomSizeND)
        iotaFixedChunkSizeND<DeviceBuf>);

    auto runTest = [&](auto&& function)
    {
        hostDeviceBuffer.getDeviceBuffer().setValue(0u);
        function(hostDeviceBuffer.getDeviceBuffer());
        hostDeviceBuffer.deviceToHost();
        validate(hostDeviceBuffer.getHostBuffer(), referenceBuffer);
    };

    // execute all tests
    std::apply([&](auto&... x) { (..., runTest(x)); }, testsFunctions);
}
