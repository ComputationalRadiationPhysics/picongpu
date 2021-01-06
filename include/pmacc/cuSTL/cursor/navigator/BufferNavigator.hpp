/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#include "tag.hpp"
#include "CartNavigator.hpp"
#include "pmacc/cuSTL/cursor/traits.hpp"

#include "pmacc/math/Vector.hpp"

#include <boost/type_traits/remove_pointer.hpp>


namespace pmacc
{
    namespace cursor
    {
        template<int T_dim>
        class BufferNavigator
        {
        public:
            typedef tag::BufferNavigator tag;
            static constexpr int dim = T_dim;

        private:
            math::Size_t<dim - 1> pitch;

        public:
            HDINLINE
            BufferNavigator(math::Size_t<dim - 1> pitch) : pitch(pitch)
            {
            }

            template<typename Data>
            HDINLINE Data operator()(const Data& data, const math::Int<dim>& jump) const
            {
                char* result = (char*) data;
                result += jump.x() * sizeof(typename boost::remove_pointer<Data>::type);
                for(int i = 1; i < dim; i++)
                    result += jump[i] * this->pitch[i - 1];
                return (Data) result;
            }

            HDINLINE
            const math::Size_t<dim - 1>& getPitch() const
            {
                return pitch;
            }
        };

        template<>
        class BufferNavigator<1>
        {
        public:
            typedef tag::BufferNavigator tag;
            static constexpr int dim = 1;

        public:
            HDINLINE
            BufferNavigator(math::Size_t<dim - 1>)
            {
            }

            template<typename Data>
            HDINLINE Data operator()(const Data& data, const math::Int<dim>& jump) const
            {
                char* result = (char*) data;
                result += jump.x() * sizeof(typename boost::remove_pointer<Data>::type);
                return (Data) result;
            }
        };

        namespace traits
        {
            template<int T_dim>
            struct dim<BufferNavigator<T_dim>>
            {
                static constexpr int value = T_dim;
            };

        } // namespace traits

    } // namespace cursor
} // namespace pmacc
