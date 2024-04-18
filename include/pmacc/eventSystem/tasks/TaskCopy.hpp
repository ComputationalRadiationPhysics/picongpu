/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
#include "pmacc/types.hpp"


namespace pmacc
{
    template<typename T_SrcBuffer, typename T_DestBuffer>
    class TaskCopy : public StreamTask
    {
        static_assert(std::is_same_v<typename T_SrcBuffer::DataBoxType, typename T_DestBuffer::DataBoxType>);

    public:
        TaskCopy(T_SrcBuffer& src, T_DestBuffer& dst) : StreamTask(), source(&src), destination(&dst)
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
