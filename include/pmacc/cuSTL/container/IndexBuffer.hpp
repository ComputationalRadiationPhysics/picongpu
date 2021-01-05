/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include <stdint.h>
#include "../vector/UInt32.hpp"
#include "../vector/Int.hpp"
#include "../vector/Size_t.hpp"
#include "../cursor/Cursor.hpp"
#include "../cursor/navigator/CartNavigator.hpp"
#include "../cursor/accessor/MarkerAccessor.hpp"
#include "../zone/SphericZone.hpp"

namespace pmacc
{
    namespace container
    {
        template<int dim>
        class IndexBuffer
        {
        private:
            math::UInt32<dim> _size;

        public:
            IndexBuffer(const math::UInt32<dim>& _size) : _size(_size)
            {
            }
            IndexBuffer(uint32_t x) : _size(x)
            {
            }
            IndexBuffer(uint32_t x, uint32_t y) : _size(x, y)
            {
            }
            IndexBuffer(uint32_t x, uint32_t y, uint32_t z) : _size(x, y, z)
            {
            }

            inline cursor::Cursor<cursor::MarkerAccessor<math::Int<dim>>, cursor::CartNavigator<dim>, math::Int<dim>>
            origin() const
            {
                math::Int<dim> factor;
                factor[0] = 1;
                factor[1] = this->_size.x();
                if(dim == 3)
                    factor[2] = this->_size.x() * this->_size.y();

                return cursor::
                    Cursor<cursor::MarkerAccessor<math::Int<dim>>, cursor::CartNavigator<dim>, math::Int<dim>>(
                        cursor::MarkerAccessor<math::Int<dim>>(),
                        cursor::CartNavigator<dim>(factor),
                        math::Int<dim>(0));
            }
            inline cursor::Cursor<cursor::MarkerAccessor<math::Int<dim>>, cursor::CartNavigator<dim>, math::Int<dim>>
            originCustomAxes(const math::UInt32<dim>& axes) const
            {
                math::Int<dim> factor;
                factor[0] = 1;
                factor[1] = this->_size.x();
                if(dim == 3)
                    factor[2] = this->_size.x() * this->_size.y();
                math::Int<dim> customFactor;
                for(uint32_t i = 0; i < dim; i++)
                    customFactor[i] = factor[axes[i]];

                return cursor::
                    Cursor<cursor::MarkerAccessor<math::Int<dim>>, cursor::CartNavigator<dim>, math::Int<dim>>(
                        cursor::MarkerAccessor<math::Int<dim>>(),
                        cursor::CartNavigator<dim>(customFactor),
                        math::Int<dim>(0));
            }
            inline zone::SphericZone<dim> zone() const
            {
                return zone::SphericZone<dim>((math::Size_t<dim>) this->_size);
            }
        };

    } // namespace container
} // namespace pmacc
