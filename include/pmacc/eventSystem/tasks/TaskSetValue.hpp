/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
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
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/eventSystem/tasks/StreamTask.hpp"
#include "pmacc/nvidia/gpuEntryFunction.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits.hpp>


namespace pmacc
{
    namespace taskSetValueHelper
    {
        /** define access operation for non-pointer types
         */
        template<typename T_Type, bool isPointer>
        struct Value
        {
            typedef const T_Type type;

            HDINLINE type& operator()(type& v) const
            {
                return v;
            }
        };

        /** define access operation for pointer types
         *
         * access first element of a pointer
         */
        template<typename T_Type>
        struct Value<T_Type, true>
        {
            typedef const T_Type PtrType;
            typedef const typename boost::remove_pointer<PtrType>::type type;

            HDINLINE type& operator()(PtrType v) const
            {
                return *v;
            }
        };

        /** Get access to a value from a pointer or reference with the same method
         */
        template<typename T_Type>
        HDINLINE typename Value<T_Type, boost::is_pointer<T_Type>::value>::type& getValue(T_Type& value)
        {
            typedef Value<T_Type, boost::is_pointer<T_Type>::value> Functor;
            return Functor()(value);
        }

    } // namespace taskSetValueHelper

    /** set a value to all elements of a box
     *
     * @tparam T_numWorkers number of workers
     * @tparam T_xChunkSize number of elements in x direction to prepare with one cupla block
     */
    template<uint32_t T_numWorkers, uint32_t T_xChunkSize>
    struct KernelSetValue
    {
        /** set value to all elements
         *
         * @tparam T_DataBox pmacc::DataBox, type of the memory box
         * @tparam T_ValueType type of the value
         * @tparam T_SizeVecType pmacc::math::Vector, index type
         * @tparam T_Acc alpaka accelerator type
         *
         * @param memBox box of which all elements shall be set to value
         * @param value value to set to all elements of memBox
         * @param size extents of memBox
         */
        template<typename T_DataBox, typename T_ValueType, typename T_SizeVecType, typename T_Acc>
        DINLINE void operator()(
            T_Acc const& acc,
            T_DataBox& memBox,
            T_ValueType const& value,
            T_SizeVecType const& size) const
        {
            using namespace mappings::threads;
            using SizeVecType = T_SizeVecType;

            SizeVecType const blockIndex(cupla::blockIdx(acc));
            SizeVecType blockSize(SizeVecType::create(1));
            blockSize.x() = T_xChunkSize;

            constexpr uint32_t numWorkers = T_numWorkers;
            uint32_t const workerIdx = cupla::threadIdx(acc).x;

            ForEachIdx<IdxConfig<T_xChunkSize, numWorkers>>{workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                auto virtualWorkerIdx(SizeVecType::create(0));
                virtualWorkerIdx.x() = linearIdx;

                SizeVecType const idx(blockSize * blockIndex + virtualWorkerIdx);
                if(idx.x() < size.x())
                    memBox(idx) = taskSetValueHelper::getValue(value);
            });
        }
    };

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    /** Set all cells of a GridBuffer on the device to a given value
     *
     * T_ValueType  = data type (e.g. float, float2)
     * T_dim   = dimension of the GridBuffer
     * T_isSmallValue = true if T_ValueType can be send via kernel parameter (on cupla T_ValueType must be smaller than
     * 256 byte)
     */
    template<class T_ValueType, unsigned T_dim, bool T_isSmallValue>
    class TaskSetValue;

    template<class T_ValueType, unsigned T_dim>
    class TaskSetValueBase : public StreamTask
    {
    public:
        typedef T_ValueType ValueType;
        static constexpr uint32_t dim = T_dim;

        TaskSetValueBase(DeviceBuffer<ValueType, dim>& dst, const ValueType& value) : StreamTask(), value(value)
        {
            this->destination = &dst;
        }

        virtual ~TaskSetValueBase()
        {
            notify(this->myId, SETVALUE, nullptr);
        }

        virtual void init() = 0;

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*)
        {
        }

    protected:
        std::string toString()
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
        typedef T_ValueType ValueType;
        static constexpr uint32_t dim = T_dim;

        TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
            : TaskSetValueBase<ValueType, dim>(dst, value)
        {
        }

        virtual ~TaskSetValue()
        {
        }

        virtual void init()
        {
            // number of elements in destination
            size_t const current_size = this->destination->getCurrentSize();
            // n-dimensional size of destination based on `current_size`
            DataSpace<dim> const area_size(this->destination->getCurrentDataSpace(current_size));

            if(area_size.productOfComponents() != 0)
            {
                auto gridSize = area_size;

                /* number of elements in x direction used to chunk the destination buffer
                 * for block parallel processing
                 */
                constexpr uint32_t xChunkSize = 256;
                constexpr uint32_t numWorkers = traits::GetNumWorkers<xChunkSize>::value;

                // number of blocks in x direction
                gridSize.x() = ceil(static_cast<double>(gridSize.x()) / static_cast<double>(xChunkSize));

                auto destBox = this->destination->getDataBox();
                CUPLA_KERNEL(KernelSetValue<numWorkers, xChunkSize>)
                (gridSize.toDim3(), numWorkers, 0, this->getCudaStream())(destBox, this->value, area_size);
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
        typedef T_ValueType ValueType;
        static constexpr uint32_t dim = T_dim;

        TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
            : TaskSetValueBase<ValueType, dim>(dst, value)
            , valuePointer_host(nullptr)
        {
        }

        virtual ~TaskSetValue()
        {
            if(valuePointer_host != nullptr)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFreeHost(valuePointer_host));
                valuePointer_host = nullptr;
            }
        }

        void init()
        {
            size_t current_size = this->destination->getCurrentSize();
            const DataSpace<dim> area_size(this->destination->getCurrentDataSpace(current_size));
            if(area_size.productOfComponents() != 0)
            {
                auto gridSize = area_size;

                /* number of elements in x direction used to chunk the destination buffer
                 * for block parallel processing
                 */
                constexpr int xChunkSize = 256;
                constexpr uint32_t numWorkers = traits::GetNumWorkers<xChunkSize>::value;

                // number of blocks in x direction
                gridSize.x() = ceil(static_cast<double>(gridSize.x()) / static_cast<double>(xChunkSize));

                ValueType* devicePtr = this->destination->getPointer();

                CUDA_CHECK(cuplaMallocHost((void**) &valuePointer_host, sizeof(ValueType)));
                *valuePointer_host = this->value; // copy value to new place

                CUDA_CHECK(cuplaMemcpyAsync(
                    devicePtr,
                    valuePointer_host,
                    sizeof(ValueType),
                    cuplaMemcpyHostToDevice,
                    this->getCudaStream()));

                auto destBox = this->destination->getDataBox();
                CUPLA_KERNEL(KernelSetValue<numWorkers, xChunkSize>)
                (gridSize.toDim3(), numWorkers, 0, this->getCudaStream())(destBox, devicePtr, area_size);
            }

            this->activate();
        }

    private:
        ValueType* valuePointer_host;
    };

} // namespace pmacc
