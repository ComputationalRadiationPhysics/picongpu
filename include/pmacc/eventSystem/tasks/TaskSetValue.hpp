/* Copyright 2013-2023 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/StreamTask.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/lockstep/BlockCfg.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"

#include <type_traits>


namespace pmacc
{
    /** set a value to all elements of a box
     *
     * @tparam T_xChunkSize number of elements in x direction to prepare with one alpaka block
     */
    template<uint32_t T_xChunkSize>
    struct KernelSetValue
    {
        /** set value to all elements
         *
         * @tparam T_DataBox pmacc::DataBox, type of the memory box
         * @tparam T_ValueType type of the value
         * @tparam T_SizeVecType pmacc::math::Vector, index type
         * @tparam T_Acc alpaka accelerator type
         * @tparam T_BlockCfg lockstep worker configuration type
         *
         * @param memBox box of which all elements shall be set to value
         * @param value value to set to all elements of memBox
         * @param size extents of memBox
         */
        template<typename T_DataBox, typename T_ValueType, typename T_SizeVecType, typename T_Acc, typename T_BlockCfg>
        DINLINE void operator()(
            T_Acc const& acc,
            T_DataBox memBox,
            T_ValueType const& value,
            T_SizeVecType const& size,
            T_BlockCfg const& blockCfg) const
        {
            using SizeVecType = T_SizeVecType;

            SizeVecType const blockIndex(device::getBlockIdx(acc));
            SizeVecType blockSize(SizeVecType::create(1));
            blockSize.x() = T_xChunkSize;

            lockstep::makeForEach<T_xChunkSize>(blockCfg.getWorker(acc))(
                [&](uint32_t const linearIdx)
                {
                    auto virtualWorkerIdx(SizeVecType::create(0));
                    virtualWorkerIdx.x() = linearIdx;

                    SizeVecType const idx(blockSize * blockIndex + virtualWorkerIdx);
                    if(idx.x() < size.x())
                    {
                        constexpr bool isPointer = std::is_pointer_v<T_ValueType>;
                        if constexpr(isPointer)
                            memBox(idx) = *value;
                        else
                            memBox(idx) = value;
                    }
                });
        }
    };

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    /** Set all cells of a GridBuffer on the device to a given value
     *
     * T_ValueType  = data type (e.g. float, float2)
     * T_dim   = dimension of the GridBuffer
     * T_isLess128ByteAndTrivillyCopyable = true if T_ValueType can be send via kernel parameter (T_ValueType
     * must be <= 128 byte) and must be trivially copyable
     */
    template<
        class T_ValueType,
        unsigned T_dim,
        bool T_isLess128ByteAndTrivillayCopyable
        = sizeof(T_ValueType) <= 128 && std::is_trivially_copyable_v<T_ValueType>>
    class TaskSetValue;

    template<class T_ValueType, unsigned T_dim>
    class TaskSetValueBase : public StreamTask
    {
    public:
        using ValueType = T_ValueType;
        static constexpr uint32_t dim = T_dim;

        TaskSetValueBase(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
            : StreamTask()
            , destination(&dst)
            , value(value)
        {
        }

        ~TaskSetValueBase() override
        {
            notify(this->myId, SETVALUE, nullptr);
        }

        void init() override = 0;

        bool executeIntern() override
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

    protected:
        std::string toString() override
        {
            return "TaskSetValue";
        }

        DeviceBuffer<ValueType, dim>* destination;
        ValueType value;
    };

    /** implementation for small values (<= 256byte)
     */
    template<class T_ValueType, unsigned T_dim>
    class TaskSetValue<T_ValueType, T_dim, true> : public TaskSetValueBase<T_ValueType, T_dim>
    {
    public:
        using ValueType = T_ValueType;
        static constexpr uint32_t dim = T_dim;

        TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
            : TaskSetValueBase<ValueType, dim>(dst, value)
        {
        }

        ~TaskSetValue() override = default;

        void init() override
        {
            // number of elements in destination
            size_t const current_size = this->destination->getCurrentSize();
            // n-dimensional size of destination based on `current_size`
            MemSpace<dim> const area_size(this->destination->getCurrentDataSpace(current_size));

            if(area_size.productOfComponents() != 0)
            {
                auto gridSize = area_size;

                /* number of elements in x direction used to chunk the destination buffer
                 * for block parallel processing
                 */
                constexpr uint32_t xChunkSize = 256;

                // number of blocks in x direction
                gridSize.x() = ceil(static_cast<double>(gridSize.x()) / static_cast<double>(xChunkSize));

                auto blockCfg = lockstep::makeBlockCfg<xChunkSize>();
                auto destBox = this->destination->getDataBox();
                auto blockSize = DataSpace<dim>::create(1);
                blockSize.x() = blockCfg.numWorkers();

                auto one = DataSpace<dim>::create(1);
                auto workDiv = alpaka::WorkDivMembers<AlpakaDim<dim>, IdxType>{
                    gridSize.toAlpakaKernelVec(),
                    blockSize.toAlpakaKernelVec(),
                    one.toAlpakaKernelVec()};
                auto kernel = alpaka::createTaskKernel<Acc<dim>>(
                    workDiv,
                    KernelSetValue<xChunkSize>{},
                    destBox,
                    this->value,
                    area_size,
                    blockCfg);
                auto queue = this->getCudaStream();
                alpaka::enqueue(queue, kernel);
            }
            this->activate();
        }
    };

    /** implementation for big values (>256 byte)
     *
     * This class uses CUDA memcopy to copy an instance of T_ValueType to the GPU
     * and runs a kernel which assigns this value to all cells.
     */
    template<class T_ValueType, unsigned T_dim>
    class TaskSetValue<T_ValueType, T_dim, false> : public TaskSetValueBase<T_ValueType, T_dim>
    {
    public:
        using ValueType = T_ValueType;
        static constexpr uint32_t dim = T_dim;

        using ValueBufferType = ::alpaka::Buf<HostDevice, T_ValueType, AlpakaDim<DIM1>, MemIdxType>;

        TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
            : TaskSetValueBase<ValueType, dim>(dst, value)
            , valueBuffer(std::make_shared<ValueBufferType>(alpaka::allocMappedBufIfSupported<ValueType, MemIdxType>(
                  manager::Device<HostDevice>::get().current(),
                  manager::Device<ComputeDevice>::get().getPlatform(),
                  MemSpace<DIM1>(1).toAlpakaMemVec())))
        {
        }

        ~TaskSetValue() override
        {
        }

        void init() override
        {
            size_t current_size = this->destination->getCurrentSize();
            const MemSpace<dim> area_size(this->destination->getCurrentDataSpace(current_size));
            if(area_size.productOfComponents() != 0)
            {
                auto gridSize = area_size;

                /* number of elements in x direction used to chunk the destination buffer
                 * for block parallel processing
                 */
                constexpr int xChunkSize = 256;

                // number of blocks in x direction
                gridSize.x() = ceil(static_cast<double>(gridSize.x()) / static_cast<double>(xChunkSize));

                auto firstElemBuffer = this->destination->as1DBufferNElem(1);
                alpaka::getPtrNative(*valueBuffer)[0] = this->value; // copy value to new place

                auto queue = this->getCudaStream();
                alpaka::memcpy(queue, firstElemBuffer, *valueBuffer, MemSpace<DIM1>(1).toAlpakaMemVec());

                auto blockCfg = lockstep::makeBlockCfg<xChunkSize>();
                auto destBox = this->destination->getDataBox();
                auto blockSize = DataSpace<dim>::create(1);
                blockSize.x() = blockCfg.numWorkers();

                auto one = DataSpace<dim>::create(1);
                auto workDiv = alpaka::WorkDivMembers<AlpakaDim<dim>, IdxType>{
                    gridSize.toAlpakaKernelVec(),
                    blockSize.toAlpakaKernelVec(),
                    one.toAlpakaKernelVec()};
                auto kernel = alpaka::createTaskKernel<Acc<dim>>(
                    workDiv,
                    KernelSetValue<xChunkSize>{},
                    destBox,
                    alpaka::getPtrNative(firstElemBuffer),
                    area_size,
                    blockCfg);
                alpaka::enqueue(queue, kernel);
            }

            this->activate();
        }

    private:
        std::shared_ptr<ValueBufferType> valueBuffer;
    };

} // namespace pmacc
