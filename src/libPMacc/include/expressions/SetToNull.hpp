/**
 * Copyright 2015 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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


#pragma once

namespace PMacc
{
namespace expressions
{

/** set pointer to NULL
 *
 * only allowed to use with a pointer
 */
struct SetToNull
{
    /** set pointer to NULL
     *
     * @tparam T_Type any type/class
     *
     * @param ptr pointer that should used
     */
    template<typename T_Type>
    HDINLINE void operator()( T_Type& ptr) const
    {
        ptr = NULL;
    }
};

}//namespace expressions
}//namespace PMacc
