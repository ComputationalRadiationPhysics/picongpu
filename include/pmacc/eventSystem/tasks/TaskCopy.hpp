/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    template<typename T_SrcBuffer, typename T_DestBuffer>
    class TaskCopy : public DeviceTask
    {
        static_assert(std::is_same_v<typename T_SrcBuffer::DataBoxType, typename T_DestBuffer::DataBoxType>);

    public:
        TaskCopy(T_SrcBuffer& src, T_DestBuffer& dst) : DeviceTask(), source(&src), destination(&dst)
        {
        }

        ~TaskCopy()
        {
            notify(this->myId, COPY, nullptr);
        }

        bool executeIntern() override
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        void init() override
        {
            /* @attention: `setSize()` must be called before `getAlpakaQueue()` is called else `setSize()`
             * is not handled as part of this task. The reason for this is that is not registered to the eventsystem
             * before `init()` is finished.
             */
            if(source->isContiguous() && destination->isContiguous())
            {
                auto src = source->as1DBuffer();
                // no need to call methods of the PMacc buffer again which will only trigger the event system and is
                // increasing the latency
                auto size = alpaka::getExtents(src);
                destination->setSize(size[0]);
                auto queue = this->getAlpakaQueue();
                alpaka::memcpy(queue, destination->as1DBuffer(), src, size);
            }
            else
            {
                size_t currentSize = source->size();
                destination->setSize(currentSize);
                auto sizeND = source->sizeND(currentSize);
                auto queue = this->getAlpakaQueue();
                alpaka::memcpy(queue, destination->getAlpakaView(), source->getAlpakaView(), sizeND.toAlpakaMemVec());
            }

            this->activate();
        }

        std::string toString() override
        {
            return "TaskCopy";
        }

    protected:
        T_SrcBuffer* source;
        T_DestBuffer* destination;
    };

} // namespace pmacc
