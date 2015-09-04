/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "eventSystem/streams/EventStream.hpp"
#include "types.h"
#include "Environment.def"

#include <cuda_runtime.h>

#include <string>
#include <stdexcept>
#include <vector>

namespace PMacc
{
    /**
     * Manages a pool of EventStreams and gives access to them.
     * This class is a singleton.
     */
    class StreamController
    {
    public:

        /**
         * Returns a pointer to the next EventStream in the controller's queue.
         * @return pointer to next EventStream
         */
        EventStream* getNextStream()
        {
            if(!isActivated)
                throw std::runtime_error(std::string("StreamController is not activated but getNextStream() was called"));
            size_t oldIndex = currentStreamIndex;
            currentStreamIndex++;
            if (currentStreamIndex == streams.size())
                currentStreamIndex = 0;

            return streams[oldIndex];
        }

        /**
         * Destructor.
         * Deletes internal streams. Tears down CUDA.
         */
        virtual ~StreamController()
        {

            for (size_t i = 0; i < streams.size(); i++)
            {
                __delete(streams[i]);
            }
            streams.clear();

            /* This is the single point in PIC where ALL CUDA work must be finished. */
            /* Accessing CUDA objects after this point may fail! */
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaDeviceReset());
        }

        /**
         * Add additional EventStreams to the queue.
         * @param count number of EventStreams to add.
         */
        void addStreams(size_t count)
        {
            for (size_t i = 0; i < count; i++)
            {
                streams.push_back(new EventStream());
            }
        }

        /** enable StreamController and add one stream
         *
         * If StreamController is not activated getNextStream() will crash on its first call
         */
        void activate()
        {
            addStreams(1);
            isActivated=true;
        }

        /**
         * Returns the number of available EventStreams in the queue.
         * @return number of EventStreams
         */
        size_t getStreamsCount()
        {
            return streams.size();
        }

    private:

        friend class Environment<DIM1>;
        friend class Environment<DIM2>;
        friend class Environment<DIM3>;

        /**
         * Constructor.
         */
        StreamController() : isActivated(false),currentStreamIndex(0)
        {
        }

        /**
         * Get instance of this class.
         * This class is a singleton class.
         * @return an instance
         */
        static StreamController& getInstance()
        {
            static StreamController instance;
            return instance;
        }

        std::vector<EventStream*> streams;
        size_t currentStreamIndex;
        bool isActivated;

    };

} //namespace PMacc
