/* Copyright 2017-2021 Heiko Burau
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


namespace pmacc
{
    namespace algorithm
    {
        namespace functor
        {
            template<typename T_Type>
            struct GetComponent
            {
                using Type = T_Type;
                using result_type = Type;
                uint32_t m_component;

                HDINLINE GetComponent(uint32_t const component) : m_component(component)
                {
                }

                template<typename Array, typename T_Acc>
                HDINLINE Type& operator()(T_Acc const&, Array& array) const
                {
                    return array[m_component];
                }

                template<typename Array>
                HDINLINE Type& operator()(Array& array) const
                {
                    return array[m_component];
                }
            };

        } // namespace functor
    } // namespace algorithm
} // namespace pmacc
