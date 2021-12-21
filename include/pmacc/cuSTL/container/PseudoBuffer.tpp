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

namespace pmacc
{
    namespace container
    {
        template<typename Type, int dim>
        template<typename _Type>
        PseudoBuffer<Type, dim>::PseudoBuffer(pmacc::DeviceBuffer<_Type, dim>& devBuffer)
        {
            cuplaPitchedPtr cuplaData = devBuffer.getCudaPitched();
            this->dataPointer = (Type*) cuplaData.ptr;
            this->_size = (math::Size_t<dim>) devBuffer.getDataSpace();
            if(dim == 2)
                this->pitch[0] = cuplaData.pitch;
            if(dim == 3)
            {
                this->pitch[0] = cuplaData.pitch;
                this->pitch[1] = cuplaData.pitch * this->_size.y();
            }
        }

        template<typename Type, int dim>
        template<typename _Type>
        PseudoBuffer<Type, dim>::PseudoBuffer(pmacc::HostBuffer<_Type, dim>& hostBuffer)
        {
            this->dataPointer = (Type*) hostBuffer.getBasePointer();
            this->_size = (math::Size_t<dim>) hostBuffer.getDataSpace();
            if(dim == 2)
                this->pitch[0] = sizeof(Type) * this->_size[0];
            if(dim == 3)
            {
                this->pitch[0] = sizeof(Type) * this->_size[0];
                this->pitch[1] = this->pitch[0] * this->_size[1];
            }
        }

    } // namespace container
} // namespace pmacc
