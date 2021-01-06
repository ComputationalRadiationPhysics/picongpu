/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
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

#include <cstdint>


namespace pmacc
{
    namespace eventSystem
    {
        /**
         * Internal event/task type used for notifications in the event system.
         */
        enum EventType
        {
            FINISHED,
            COPYHOST2DEVICE,
            COPYDEVICE2HOST,
            COPYDEVICE2DEVICE,
            SENDFINISHED,
            RECVFINISHED,
            LOGICALAND,
            SETVALUE,
            GETVALUE,
            KERNEL
        };

    } // namespace eventSystem

    // for backward compatibility pull all definitions into the pmacc namespace
    using namespace eventSystem;
} // namespace pmacc
