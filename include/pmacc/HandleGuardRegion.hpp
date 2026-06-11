/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace pmacc
{
    /**
     * Property struct that exposes policies for handling data in the guard region
     * Each policy must handle both sides of the (possible) exchange:
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
        using HandleExchanged = T_HandleExchanged;
        using HandleNotExchanged = T_HandleNotExchanged;
    };

} // namespace pmacc
