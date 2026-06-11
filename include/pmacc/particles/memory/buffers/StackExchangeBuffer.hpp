/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                (vint_t*) alpaka::getPtrNative(stack.getDeviceBuffer().sizeDeviceSideBuffer()),
                stack.getDeviceBuffer().capacityND().productOfComponents(),
                PushDataBox<vint_t, FRAMEINDEX>(
                    stackIndexer.getDeviceBuffer().data(),
                    (vint_t*) alpaka::getPtrNative(stackIndexer.getDeviceBuffer().sizeDeviceSideBuffer())));
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

        void setSize(size_t const size)
        {
            // do host and device setSize parallel
            EventTask split = eventSystem::getTransactionEvent();
            EventTask e1;

            if(!Environment<>::get().isMpiDirectEnabled())
            {
                eventSystem::startTransaction(split);
                stackIndexer.getHostBuffer().setSize(size);
                stack.getHostBuffer().setSize(size);
                e1 = eventSystem::endTransaction();
            }

            eventSystem::startTransaction(split);
            stackIndexer.getDeviceBuffer().setSize(size);
            EventTask e2 = eventSystem::endTransaction();
            eventSystem::startTransaction(split);
            stack.getDeviceBuffer().setSize(size);
            EventTask e3 = eventSystem::endTransaction();

            eventSystem::setTransactionEvent(e1 + e2 + e3);
        }

        size_t getHostCurrentSize()
        {
            size_t result = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                result = stackIndexer.getDeviceBuffer().size();
            else
                result = stackIndexer.getHostBuffer().size();

            return result;
        }

        size_t getDeviceCurrentSize()
        {
            return stackIndexer.getDeviceBuffer().size();
        }

        size_t getDeviceParticlesCurrentSize()
        {
            return stack.getDeviceBuffer().size();
        }

        size_t getHostParticlesCurrentSize()
        {
            if(Environment<>::get().isMpiDirectEnabled())
                return stack.getDeviceBuffer().size();

            return stack.getHostBuffer().size();
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
