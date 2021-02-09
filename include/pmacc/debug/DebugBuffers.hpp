/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/types.hpp"

#include <string>
#include <sstream>

namespace pmacc
{
    /**
     * Helper class for debugging buffers
     *
     * @tparam DIM dimension of the buffer to debug.
     */
    template<unsigned DIM>
    class DebugBuffers
    {
    public:
        /**
         * Converts a HostBuffer to a string for debugging.
         *
         * @tparam TYPE datatype stored in the buffer
         * @param hostBuffer the HostBuffer to convert to a string
         * @return a string representing the buffer
         */
        template<class TYPE>
        static std::string bufferToStr(HostBuffer<TYPE, DIM>& hostBuffer);
    };

    template<>
    class DebugBuffers<DIM2>
    {
    public:
        template<class TYPE>
        static std::string bufferToStr(HostBuffer<TYPE, DIM2>& hostBuffer)
        {
            std::stringstream stream;

            typename HostBuffer<TYPE, DIM2>::DataBoxType db = hostBuffer.getDataBox();

            for(size_t y = 0; y < hostBuffer.getDataSpace().y(); y++)
            {
                for(size_t x = 0; x < hostBuffer.getDataSpace().x(); x++)
                    stream << db[y][x] << " ";

                stream << std::endl;
            }

            return stream.str();
        }
    };

    template<>
    class DebugBuffers<DIM3>
    {
    public:
        template<class TYPE>
        static std::string bufferToStr(HostBuffer<TYPE, DIM3>& hostBuffer)
        {
            std::stringstream stream;

            typename HostBuffer<TYPE, DIM3>::DataBoxType db = hostBuffer.getDataBox();

            for(size_t z = 0; z < hostBuffer.getDataSpace().z(); z++)
            {
                stream << "z = " << z << std::endl;

                for(size_t y = 0; y < hostBuffer.getDataSpace().y(); y++)
                {
                    for(size_t x = 0; x < hostBuffer.getDataSpace().x(); x++)
                        stream << db[z][y][x] << " ";

                    stream << std::endl;
                }
            }

            return stream.str();
        }
    };
} // namespace pmacc
