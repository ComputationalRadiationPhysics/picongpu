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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/eventSystem/events/CudaEventHandle.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    /** Wrapper for a single alpaka queue.
     *
     * Allows recording alpaka events on the queue.
     */
    class EventStream
    {
    public:
        EventStream();

        /** Destructor.
         *
         * Waits for the stream to finish and destroys it.
         */
        virtual ~EventStream();

        /** Returns the alpaka queue object associated with this EventStream.
         *
         * @return the internal alpaka queue object
         */
        AccStream getCudaStream() const;

        void waitOn(const CudaEventHandle& ev);

    private:
        AccStream stream;
    };

} // namespace pmacc
