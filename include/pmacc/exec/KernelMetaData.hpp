/* Copyright 2022 Rene Widera
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

#include "pmacc/types.hpp"

#include <string>


namespace pmacc::exec::detail
{
    //! Meta data of a device kernel
    class KernelMetaData
    {
        //! file name
        std::string const m_file;
        //! line number
        size_t const m_line;

    public:
        KernelMetaData(std::string const& file, size_t const line) : m_file(file), m_line(line)
        {
        }

        //! file name from where the kernel is called
        std::string getFile() const
        {
            return m_file;
        }

        //! line number in the file where the kernel is called
        size_t getLine() const
        {
            return m_line;
        }
    };
} // namespace pmacc::exec::detail
