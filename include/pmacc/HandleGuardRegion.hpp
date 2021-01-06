/* Copyright 2015-2021 Alexander Grund
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

namespace pmacc
{
    /**
     * Property struct that exposes policies for handling data in the guard region
     * Each police must handle both sides of the (possible) exchange:
     *      - Outgoing side: E.g. particles going out of the local volume
     *      - Incoming side: E.g. particles coming into the local volume
     *
     * All policies have the functions _handleOutgoing_ and _handleIncoming_
     * with signature void(TypeOfData&, int32_t direction)
     *
     * @tparam T_HandleExchanged Policy for handling data that should be exchanged
     *         with a neighboring rank
     * @tparam T_HandleLostParticles Policy for handling data that is not sent/received
     *         to/from any other rank, which is the case for the boundary of the total
     *         volume when non-periodic conditions are used
     */
    template<class T_HandleExchanged, class T_HandleNotExchanged>
    struct HandleGuardRegion
    {
        typedef T_HandleExchanged HandleExchanged;
        typedef T_HandleNotExchanged HandleNotExchanged;
    };

} // namespace pmacc
