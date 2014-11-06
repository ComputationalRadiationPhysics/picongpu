/**
 * Copyright 2013 Felix Schmitt, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#ifndef DEBUGEXCHANGETYPES_HPP
#define	DEBUGEXCHANGETYPES_HPP

#include <string>
#include <sstream>

#include "types.h"
#include "memory/dataTypes/Mask.hpp"

namespace PMacc
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

            if (mask.containsExchangeType(LEFT))
                stream << "LEFT ";

            if (mask.containsExchangeType(RIGHT))
                stream << "RIGHT ";

            if (mask.containsExchangeType(TOP))
                stream << "TOP ";

            if (mask.containsExchangeType(BOTTOM))
                stream << "BOTTOM ";

            if (mask.containsExchangeType(FRONT))
                stream << "FRONT ";

            if (mask.containsExchangeType(BACK))
                stream << "BACK ";

            stream << "]";

            return stream.str();
        }
    };

}

#endif	/* DEBUGEXCHANGETYPES_HPP */

