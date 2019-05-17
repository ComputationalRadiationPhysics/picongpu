/* Copyright 2015-2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

    template<
        typename T_Type,
        size_t T_size
    >
    struct Array{
        T_Type m_data[T_size];

        template<
            typename T_Idx
        >
        ALPAKA_FN_HOST_ACC
        const T_Type &
        operator[](
            const T_Idx idx
        ) const {
            return m_data[idx];
        }

        template<
            typename T_Idx
        >
        ALPAKA_FN_HOST_ACC
        T_Type &
        operator[](
            const T_Idx idx
        ){
            return m_data[idx];
        }
    };

} //namespace CUPLA_ACCELERATOR_NAMESPACE
} //namespace cupla
