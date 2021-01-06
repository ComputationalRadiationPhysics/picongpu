/* Copyright 2013-2021 Rene Widera
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

#include <boost/mpl/pair.hpp>


namespace pmacc
{
    /** Wrapper to use any type as identifier
     *
     * Wrap a type thus we can call default constructor on every class
     * This is needed to support that any type can used as identifier in for math::MapTuple
     */
    template<typename T_Type>
    struct TypeAsIdentifier
    {
        typedef T_Type type;
    };

    /** Unary functor to wrap any type with TypeAsIdentifier
     *
     * @tparam T_Type to to wrap
     */
    template<typename T_Type>
    struct MakeIdentifier
    {
        typedef TypeAsIdentifier<T_Type> type;
    };

    /** Pass through of an already existing Identifier
     *
     * Avoids double-wrapping of an Identifier
     */
    template<typename T_Type>
    struct MakeIdentifier<TypeAsIdentifier<T_Type>>
    {
        typedef TypeAsIdentifier<T_Type> type;
    };

    /** create boost mpl pair <TypeAsIdentifier<Type>,PointerOfType>
     *
     * @tparam T_Type any type
     * @return ::type boost::mpl::pair<TypeAsIdentifier<Type>,PointerOfType>
     */
    template<typename T_Type>
    struct TypeToPointerPair
    {
        typedef T_Type* TypePtr;
        typedef bmpl::pair<typename MakeIdentifier<T_Type>::type, TypePtr> type;
    };

} // namespace pmacc
