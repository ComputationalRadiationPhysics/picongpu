/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <string>
#include <sstream>

namespace pmacc
{
    /**
     * Helper class for debugging exchange types.
     *
     */
    class DebugExchangeTypes
    {
    public:
        /**
         * Converts an exchange type to a string for debugging.
         *
         * @param exchangeType the exchange type to convert
         * @return a string representing the exchange type
         */
        static std::string exchangeTypeToString(uint32_t exchangeType)
        {
            Mask mask(exchangeType);

            std::stringstream stream;
            stream << "[";

            if(mask.containsExchangeType(LEFT))
                stream << "LEFT ";

            if(mask.containsExchangeType(RIGHT))
                stream << "RIGHT ";

            if(mask.containsExchangeType(TOP))
                stream << "TOP ";

            if(mask.containsExchangeType(BOTTOM))
                stream << "BOTTOM ";

            if(mask.containsExchangeType(FRONT))
                stream << "FRONT ";

            if(mask.containsExchangeType(BACK))
                stream << "BACK ";

            stream << "]";

            return stream.str();
        }
    };

} // namespace pmacc
