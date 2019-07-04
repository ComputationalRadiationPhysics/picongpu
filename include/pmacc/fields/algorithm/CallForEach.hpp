/* Copyright 2019 Sergei Bastrakov
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

#include "pmacc/Environment.hpp"
#include "pmacc/fields/algorithm/ForEach.hpp"
#include "pmacc/functor/Interface.hpp"


namespace pmacc
{
namespace fields
{
namespace algorithm
{

    /// TODO
    template<
        typename T_Field,
        typename T_FunctorOperator
    >
    struct CallForEach
    {
        /// TODO
        /** Operate on the domain CORE, BORDER and GUARD
         *
         * @param currentStep current simulation time step
         */
        HINLINE void
        operator( )( uint32_t const currentStep ) const
        {
            using UnaryFunctor = functor::Interface<
                typename T_FunctorOperator::type,
                1u,
                void
            >;

            DataConnector &dc = Environment<>::get().DataConnector();
            auto & field = *dc.get< T_Field >(
                T_Field::getName( ),
                true
            );

            forEach(
                field,
                UnaryFunctor( currentStep )
            );

            dc.releaseData( T_Field::getName( ) );
        }
    };

} // namespace algorithm
} // namespace fields
} // namespace pmacc
