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
#include "pmacc/meta/conversion/TypeToPair.hpp"

namespace pmacc
{
    /** create boost mpl pair
     *
     * If T_Type is a pmacc alias than first is set to anonym alias name
     * and second is set to T_Type.
     * If T_Type is no alias than TypeToPair is used.
     *
     * @tparam T_Type any type
     * @resturn ::type
     */
    template<typename T_Type>
    struct TypeToAliasPair
    {
        typedef typename TypeToPair<T_Type>::type type;
    };

    /** specialisation if T_Type is a pmacc alias*/
    template<template<typename, typename> class T_Alias, typename T_Type>
    struct TypeToAliasPair<T_Alias<T_Type, pmacc::pmacc_isAlias>>
    {
        typedef bmpl::pair<T_Alias<pmacc_void, pmacc::pmacc_isAlias>, T_Alias<T_Type, pmacc::pmacc_isAlias>> type;
    };


} // namespace pmacc
