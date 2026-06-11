/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/dataManagement/ISimulationData.hpp"
#include "pmacc/dimensions/Definition.hpp"

#include <cstdint>
#include <memory>
#include <string>

#if (ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <memory>

#    include <mallocMC/mallocMC.hpp>

namespace pmacc
{
    template<typename T_DeviceHeap>
    class MallocMCBuffer : public ISimulationData
    {
    public:
        using DeviceHeap = T_DeviceHeap;
        using BufferType = ::alpaka::Buf<HostDevice, uint8_t, AlpakaDim<DIM1>, MemIdxType>;

        MallocMCBuffer(DeviceHeap& deviceHeap);

        virtual ~MallocMCBuffer();

        SimulationDataId getUniqueId() override
        {
            return getName();
        }

        static std::string getName()
        {
            return std::string("MallocMCBuffer");
        }

        int64_t getOffset()
        {
            return hostBufferOffset;
        }

        void synchronize() override;

    private:
        std::optional<BufferType> hostBuffer;
        int64_t hostBufferOffset;
        mallocMC::HeapInfo deviceHeapInfo;
    };


} // namespace pmacc

#    include "pmacc/particles/memory/buffers/MallocMCBuffer.tpp"

#else

namespace pmacc
{
    template<typename T_DeviceHeap>
    class MallocMCBuffer : public ISimulationData
    {
    public:
        MallocMCBuffer(T_DeviceHeap const&)
        {
        }

        ~MallocMCBuffer() override = default;

        SimulationDataId getUniqueId() override
        {
            return getName();
        }

        static std::string getName()
        {
            return std::string("MallocMCBuffer");
        }

        int64_t getOffset()
        {
            return 0u;
        }

        void synchronize() override
        {
        }
    };

} // namespace pmacc
#endif
