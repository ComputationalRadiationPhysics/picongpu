/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/Environment.def"
#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/eventSystem/queues/Queue.hpp"
#include "pmacc/types.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace pmacc
{
    /**
     * Manages a pool of ComputeDeviceQueues and gives access to them.
     * This class is a singleton.
     */
    class QueueController
    {
    public:
        /**
         * Returns a pointer to the next Queue in the controller's queue.
         * @return pointer to next Queue
         */
        Queue* getNextStream()
        {
            if(!isActivated)
                throw std::runtime_error(
                    std::string("QueueController is not activated but getNextStream() was called"));
            size_t oldIndex = currentStreamIndex;
            currentStreamIndex++;
            if(currentStreamIndex == queues.size())
                currentStreamIndex = 0;

            return queues[oldIndex].get();
        }

        /**
         * Destructor.
         * Deletes internal streams. Tears down CUDA.
         */
        virtual ~QueueController()
        {
            // First delete the streams
            queues.clear();

            /* This is the single point in PIC where ALL CUDA work must be finished. */
            /* Accessing CUDA objects after this point may fail! */
            alpaka::wait(manager::Device<ComputeDevice>::get().current());
            manager::Device<ComputeDevice>::get().reset();
        }

        /**
         * Add additional Queues to the queue.
         * @param count number of Queues to add.
         */
        void addQueues(size_t count)
        {
            for(size_t i = 0; i < count; i++)
            {
                queues.push_back(std::make_shared<Queue>());
            }
        }

        /** enable QueueController and add one stream
         *
         * If QueueController is not activated getNextStream() will crash on its first call
         */
        void activate()
        {
            addQueues(1);
            isActivated = true;
        }

        /**
         * Returns the number of available Queues in the queue.
         * @return number of Queues
         */
        size_t getStreamsCount()
        {
            return queues.size();
        }

    private:
        friend struct detail::Environment;

        /**
         * Constructor.
         */
        QueueController() = default;

        /**
         * Get instance of this class.
         * This class is a singleton class.
         * @return an instance
         */
        static QueueController& getInstance()
        {
            static QueueController instance;
            return instance;
        }

        std::vector<std::shared_ptr<Queue>> queues;
        size_t currentStreamIndex{0};
        bool isActivated{false};
    };

} // namespace pmacc
