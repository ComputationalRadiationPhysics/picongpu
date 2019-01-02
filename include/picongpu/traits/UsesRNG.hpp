/* Copyright 2016-2019 Marco Garten, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <boost/type_traits/integral_constant.hpp>

namespace picongpu
{
namespace traits
{

/** Checks if an object requires the RNG
 *
 * @tparam T_Object any object (class or typename)
 *
 * This struct must inherit from (boost::true_type/false_type)
 */
template<typename T_Object>
struct UsesRNG : public boost::false_type
{
};

}// namespace traits

}// namespace picongpu
