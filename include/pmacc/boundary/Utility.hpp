/* Copyright 2021-2023 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/type/Exchange.hpp"


namespace pmacc
{
    namespace boundary
    {
        /** Return if the given exchange is axis aligned
         *
         * Axis aligned means it is only x, only y, or only z, and not a mix of those.
         *
         * @param exchangeType number characterizing exchange @see pmacc::type::ExchangeType
         */
        HDINLINE bool isAxisAligned(uint32_t exchangeType)
        {
            return (FRONT % exchangeType == 0);
        }

        /** Return if the given axis alighed exchange is on the max (right) side on that axis
         *
         * @param exchangeType number characterizing exchange @see pmacc::type::ExchangeType
         */
        HINLINE bool isMaxSide(uint32_t exchangeType)
        {
            if(!isAxisAligned(exchangeType))
                throw std::runtime_error("isPositive() called for not axis aligned exchangeType");
            return (exchangeType % 2);
        }

        /** Return if the given axis alighed exchange is on the min (left) side on that axis
         *
         * @param exchangeType number characterizing exchange @see pmacc::type::ExchangeType
         */
        HINLINE bool isMinSide(uint32_t exchangeType)
        {
            return !isMaxSide(exchangeType);
        }

        /** Get axis (0 = x, 1 = y, 2 = z) for the given axis aligned exchange
         *
         * Throws for not axis aligned exchanges.
         *
         * @param exchangeType number characterizing exchange @see pmacc::type::ExchangeType
         */
        HINLINE uint32_t getAxis(uint32_t exchangeType)
        {
            if(!isAxisAligned(exchangeType))
                throw std::runtime_error("getAxis() called for not axis aligned exchangeType");
            uint32_t axis = 0u;
            if(exchangeType >= BOTTOM && exchangeType <= TOP)
                axis = 1u;
            if(exchangeType >= BACK)
                axis = 2u;
            return axis;
        }

    } // namespace boundary
} // namespace pmacc
