/* Copyright 2019-2020 Sergei Bastrakov
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

#include <memory>
#include <utility>


namespace pmacc
{
namespace memory
{

    /*
     * Analogue of std::make_unique for C++11, except not disabled for arrays.
     * Implementation is taken from
     * https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
     */
    template<
        typename T,
        typename ... T_Args
    >
    inline std::unique_ptr< T > makeUnique( T_Args && ... args )
    {
        return std::unique_ptr< T >( new T( std::forward< T_Args >( args ) ... ) );
    }

} // namespace memory
} // namespace pmacc
