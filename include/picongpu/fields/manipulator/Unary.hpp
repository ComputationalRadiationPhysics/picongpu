/* Copyright 2019 Sergei Bastrakov
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

#include <pmacc/functor/Interface.hpp>


namespace picongpu
{
namespace fields
{
namespace manipulator
{

    /** Interface for a unary field functor
     *
     * The user functor is passed by the manipulation algorithm
     * (e.g. picongpu::fields::Manipulate) to this interface, there is
     * no need to do this explicitly in the param files.
     *
     * @tparam T_UnaryFunctor unary field functor, must contain
     *                        `void operator()(T & fieldValue)`
     */
    template< typename T_UnaryFunctor >
    using Unary = pmacc::functor::Interface<
        T_UnaryFunctor,
        1u,
        void
    >;

} // namespace manipulator
} // namespace fields
} // namespace picongpu
