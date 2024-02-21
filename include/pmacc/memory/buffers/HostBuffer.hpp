/* Copyright 2013-2023 Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"

namespace pmacc
{
    class EventTask;

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /** N-dimensional host buffer
     *
     * @tparam T_Type datatype for buffer data
     * @tparam T_dim dimension of the buffer
     */
    template<typename T_Type, uint32_t T_dim>
    class HostBuffer : public Buffer<T_Type, T_dim>
    {
        using BufferType = ::alpaka::Buf<HostDevice, T_Type, AlpakaDim<DIM1>, MemIdxType>;
        using ViewType = alpaka::ViewPlainPtr<HostDevice, T_Type, AlpakaDim<T_dim>, MemIdxType>;

    public:
        using DataBoxType = typename Buffer<T_Type, T_dim>::DataBoxType;
        std::optional<BufferType> hostBuffer;
        std::optional<ViewType> view;

        using BufferType1D = ::alpaka::ViewPlainPtr<HostDevice, T_Type, AlpakaDim<DIM1>, MemIdxType>;

        BufferType1D as1DBuffer()
        {
            auto currentSize = this->getCurrentSize();
            eventSystem::startOperation(ITask::TASK_HOST);
            return BufferType1D(
                alpaka::getPtrNative(*view),
                alpaka::getDev(*hostBuffer),
                MemSpace<DIM1>(currentSize).toAlpakaMemVec());
        }

        ViewType getAlpakaView() const
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return *view;
        }

        /** allocate data accessible from the host
         *
         * @param size extent for each dimension (in elements)
         */
        HostBuffer(MemSpace<T_dim> size)
            : Buffer<T_Type, T_dim>(size)
            , hostBuffer(alpaka::allocMappedBufIfSupported<T_Type, MemIdxType>(
                  manager::Device<HostDevice>::get().current(),
                  manager::Device<ComputeDevice>::get().getPlatform(),
                  MemSpace<DIM1>(size.productOfComponents()).toAlpakaMemVec()))
        {
            MemSpace<T_dim> pitchInBytes;
            pitchInBytes.x() = sizeof(T_Type);
            for(uint32_t d = 1u; d < T_dim; ++d)
                pitchInBytes[d] = pitchInBytes[d - 1u] * size[d - 1u];
            view.emplace(ViewType(
                alpaka::getPtrNative(*hostBuffer),
                alpaka::getDev(*hostBuffer),
                size.toAlpakaMemVec(),
                pitchInBytes.toAlpakaMemVec()));
            reset(false);
        }

        /** create a shallow view into an existing buffer
         *
         * @param source buffer to create the view on
         * @param size extent for each dimension (in elements)
         * @param offset offset within the source (in elements)
         *
         * @attention offset + size must be leather equal to the size of the source buffer
         */
        HostBuffer(HostBuffer& source, MemSpace<T_dim> size, MemSpace<T_dim> offset = MemSpace<T_dim>())
            : Buffer<T_Type, T_dim>(size)
            , hostBuffer(source->hostBuffer)
        {
            auto subView = createSubView(*source.view, size.toAlpakaMemVec(), offset.toAlpakaMemVec());
            view.emplace(ViewType(
                alpaka::getPtrNative(subView),
                alpaka::getDev(subView),
                alpaka::getExtents(subView),
                alpaka::getPitchesInBytes(subView)));
            reset(true);
        }

        ~HostBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
        }

        /** Copies the data from the given DeviceBuffer to this HostBuffer.
         *
         * @param other DeviceBuffer to copy data from
         */
        void copyFrom(DeviceBuffer<T_Type, T_dim>& other)
        {
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        T_Type* data() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            return alpaka::getPtrNative(*view);
        }

        void reset(bool preserveData = true) override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            this->setCurrentSize(this->getDataSpace().productOfComponents());
            if(!preserveData)
            {
                /* if it is a pointer out of other memory we can not assume that
                 * that the physical memory is contiguous
                 */
                if(hostBuffer && alpaka::getPtrNative(*hostBuffer) == alpaka::getPtrNative(*view))
                    memset(
                        reinterpret_cast<void*>(alpaka::getPtrNative(*view)),
                        0,
                        this->getDataSpace().productOfComponents() * sizeof(T_Type));
                else
                {
                    // Using Array is a workaround for types without default constructor
                    memory::Array<uint8_t, sizeof(T_Type)> tmp(uint8_t{0});
                    // use first element to avoid issue because Array is aligned (sizeof can be larger than component
                    // type)
                    setValue(*reinterpret_cast<T_Type*>(tmp.data()));
                }
            }
        }

        void setValue(const T_Type& value) override
        {
            // getDataBox is notifying the event system, no need to do it manually
            auto memBox = this->getDataBox();
            auto current_size = static_cast<int64_t>(this->getCurrentSize());
            using D1Box = DataBoxDim1Access<DataBoxType>;
            D1Box d1Box(memBox, this->getDataSpace());
#pragma omp parallel for
            for(int64_t i = 0; i < current_size; i++)
            {
                d1Box[i] = value;
            }
        }

        DataBoxType getDataBox() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            auto pitchBytes = MemSpace<T_dim>(getPitchesInBytes(*view));
            return DataBoxType(PitchedBox<T_Type, T_dim>(alpaka::getPtrNative(*view), pitchBytes));
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCurrentSize() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            // getCurrentSize is notifying the event system, no need to do it manually
            size_t const size = this->getCurrentSize();
            return {alpaka::getPtrNative(*view), size};
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCapacity() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            size_t const size = this->getDataSpace().productOfComponents();
            eventSystem::startOperation(ITask::TASK_HOST);
            return {alpaka::getPtrNative(*view), size};
        }
    };

} // namespace pmacc
