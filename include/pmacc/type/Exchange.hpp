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

#include <string>


namespace pmacc
{
    namespace type
    {
        /**
         * Bitmask which describes the direction of communication.
         *
         * Bitmasks may be combined logically, e.g. LEFT+TOP = TOPLEFT.
         * It is not possible to combine complementary masks (e.g. FRONT and BACK),
         * as a bitmask always defines one direction of communication (send or receive).
         *
         * Axis index relation:
         *   right & left are in X
         *   bottom & top are in Y
         *   back & front are in Z
         */
        enum ExchangeType
        {
            RIGHT = 1u,
            LEFT = 2u,
            BOTTOM = 3u,
            TOP = 6u,
            BACK = 9u,
            FRONT = 18u // 3er-System
        };

        struct ExchangeTypeNames
        {
            std::string operator[](const uint32_t exchange) const
            {
                if(exchange >= 27)
                    return std::string("unknown exchange type: ") + std::to_string(exchange);

                const char* names[27]
                    = {"none",
                       "right",
                       "left",
                       "bottom",
                       "right-bottom",
                       "left-bottom",
                       "top",
                       "right-top",
                       "left-top",
                       "back",
                       "right-back",
                       "left-back",
                       "bottom-back",
                       "right-bottom-back",
                       "left-bottom-back",
                       "top-back",
                       "right-top-back",
                       "left-top-back",
                       "front",
                       "right-front",
                       "left-front",
                       "bottom-front",
                       "right-bottom-front",
                       "left-bottom-front",
                       "top-front",
                       "right-top-front",
                       "left-top-front"};
                return names[exchange];
            }
        };

    } // namespace type

    // for backward compatibility pull all definitions into the pmacc namespace
    using namespace type;
} // namespace pmacc
