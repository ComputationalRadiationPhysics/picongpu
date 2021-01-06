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

#include "pmacc/cuSTL/algorithm/host/Foreach.hpp"

#include <pmacc/cuSTL/algorithm/functor/AssignValue.hpp>

#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/int.hpp>

#include <stdint.h>


namespace pmacc
{
    namespace assigner
    {
        namespace bmpl = boost::mpl;

        template<typename T_Dim = bmpl::_1, typename T_CartBuffer = bmpl::_2>
        struct HostMemAssigner
        {
            static constexpr int dim = T_Dim::value;
            typedef T_CartBuffer CartBuffer;

            template<typename Type>
            HINLINE void assign(const Type& value)
            {
                // "Curiously recurring template pattern"
                CartBuffer* buffer = static_cast<CartBuffer*>(this);

                // get a host accelerator
                auto hostDev = cupla::manager::Device<cupla::AccHost>::get().device();

                algorithm::host::Foreach foreach;
                foreach(hostDev, buffer->zone(), buffer->origin(), pmacc::algorithm::functor::AssignValue<Type>(value))
                    ;
            }
        };

    } // namespace assigner
} // namespace pmacc
