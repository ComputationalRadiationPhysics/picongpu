/* Copyright 2014-2021 Felix Schmitt
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

#include <sstream>

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{
    /**
     * Any DIM-dimensional selection of a simulation volume with a size and offset.
     *
     * @tparam DIM number of dimensions
     */
    template<unsigned DIM>
    class Selection
    {
    public:
        /**
         * Constructor
         * Size and offset initialized to 0 (empty selection)
         */
        Selection(void)
        {
            for(uint32_t i = 0; i < DIM; ++i)
            {
                size[i] = 0;
                offset[i] = 0;
            }
        }

        /**
         * Copy constructor
         */
        constexpr Selection(const Selection&) = default;

        /**
         * Constructor
         * Offset is initialized to 0.
         *
         * @param size DataSpace for selection size
         */
        Selection(DataSpace<DIM> size) : size(size)
        {
            for(uint32_t i = 0; i < DIM; ++i)
            {
                offset[i] = 0;
            }
        }

        /**
         * Constructor
         *
         * @param size DataSpace for selection size
         * @param offset DataSpace for selection offset
         */
        Selection(DataSpace<DIM> size, DataSpace<DIM> offset) : size(size), offset(offset)
        {
        }

        /**
         * Return a string representation
         *
         * @return string representation
         */
        HINLINE const std::string toString(void) const
        {
            std::stringstream str;
            str << "{ size = " << size.toString() << " offset = " << offset.toString() << " }";
            return str.str();
        }

        DataSpace<DIM> size;

        DataSpace<DIM> offset;
    };

} // namespace pmacc
