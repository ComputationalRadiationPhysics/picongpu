/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/assert.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"
#include "pmacc/particles/memory/boxes/ExchangePopDataBox.hpp"
#include "pmacc/particles/memory/boxes/ExchangePushDataBox.hpp"

namespace pmacc
{
    /**
     * Can be used for creating several DataBox types from an Exchange.
     *
     * @tparam FRAME frame datatype
     */
    template<class FRAME, class FRAMEINDEX, unsigned DIM>
    class StackExchangeBuffer
    {
    public:
        /**
         * Create a stack from any ExchangeBuffer<FRAME,DIM>.
         *
         * If the stack's internal GridBuffer has no sizeOnDevice, no device querys are allowed.
         *
         * @param stack Exchange
         */
        StackExchangeBuffer(Exchange<FRAME, DIM1>& stack, Exchange<FRAMEINDEX, DIM1>& stackIndexer)
            : stack(stack)
            , stackIndexer(stackIndexer)
        {
        }

        /**
         * Returns a PopDataBox for the internal HostBuffer.
         *
         * @return PopDataBox for host buffer
         */
        ExchangePopDataBox<vint_t, FRAME, DIM> getHostExchangePopDataBox()
        {
            return ExchangePopDataBox<vint_t, FRAME, DIM>(
                stack.getHostBuffer().getDataBox(),
                stackIndexer.getHostBuffer().getDataBox());
        }

        /**
         * Returns a PushDataBox for the internal DeviceBuffer.
         *
         * @return PushDataBox for device buffer
         */
        ExchangePushDataBox<vint_t, FRAME, DIM> getDeviceExchangePushDataBox()
        {
            PMACC_ASSERT(stack.getDeviceBuffer().hasCurrentSizeOnDevice() == true);
            PMACC_ASSERT(stackIndexer.getDeviceBuffer().hasCurrentSizeOnDevice() == true);
            return ExchangePushDataBox<vint_t, FRAME, DIM>(
                stack.getDeviceBuffer().data(),
                (vint_t*) alpaka::getPtrNative(stack.getDeviceBuffer().getCurrentSizeDeviceSideBuffer()),
                stack.getDeviceBuffer().capacityND().productOfComponents(),
                PushDataBox<vint_t, FRAMEINDEX>(
                    stackIndexer.getDeviceBuffer().data(),
                    (vint_t*) alpaka::getPtrNative(stackIndexer.getDeviceBuffer().getCurrentSizeDeviceSideBuffer())));
        }

        /**
         * Returns a PopDataBox for the internal DeviceBuffer.
         *
         * @return PopDataBox for device buffer
         */
        ExchangePopDataBox<vint_t, FRAME, DIM> getDeviceExchangePopDataBox()
        {
            return ExchangePopDataBox<vint_t, FRAME, DIM>(
                stack.getDeviceBuffer().getDataBox(),
                stackIndexer.getDeviceBuffer().getDataBox());
        }

        void setCurrentSize(const size_t size)
        {
            // do host and device setCurrentSize parallel
            EventTask split = eventSystem::getTransactionEvent();
            EventTask e1;

            if(!Environment<>::get().isMpiDirectEnabled())
            {
                eventSystem::startTransaction(split);
                stackIndexer.getHostBuffer().setCurrentSize(size);
                stack.getHostBuffer().setCurrentSize(size);
                e1 = eventSystem::endTransaction();
            }

            eventSystem::startTransaction(split);
            stackIndexer.getDeviceBuffer().setCurrentSize(size);
            EventTask e2 = eventSystem::endTransaction();
            eventSystem::startTransaction(split);
            stack.getDeviceBuffer().setCurrentSize(size);
            EventTask e3 = eventSystem::endTransaction();

            eventSystem::setTransactionEvent(e1 + e2 + e3);
        }

        size_t getHostCurrentSize()
        {
            size_t result = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                result = stackIndexer.getDeviceBuffer().getCurrentSize();
            else
                result = stackIndexer.getHostBuffer().getCurrentSize();

            return result;
        }

        size_t getDeviceCurrentSize()
        {
            return stackIndexer.getDeviceBuffer().getCurrentSize();
        }

        size_t getDeviceParticlesCurrentSize()
        {
            return stack.getDeviceBuffer().getCurrentSize();
        }

        size_t getHostParticlesCurrentSize()
        {
            if(Environment<>::get().isMpiDirectEnabled())
                return stack.getDeviceBuffer().getCurrentSize();

            return stack.getHostBuffer().getCurrentSize();
        }

        size_t getMaxParticlesCount()
        {
            size_t result = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                result = stack.getDeviceBuffer().capacityND().productOfComponents();
            else
                result = stack.getHostBuffer().capacityND().productOfComponents();

            return result;
        }

    private:
        Exchange<FRAME, DIM1>& getExchangeBuffer()
        {
            return stack;
        }

        Exchange<FRAME, DIM1>& stack;
        Exchange<FRAMEINDEX, DIM1>& stackIndexer;
    };
} // namespace pmacc
