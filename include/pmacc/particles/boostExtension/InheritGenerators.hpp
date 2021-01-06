/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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
#include "pmacc/particles/memory/frames/NullFrame.hpp"

#include <boost/mpl/vector.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/empty.hpp>


namespace pmacc
{
    template<class list_>
    struct LinearInherit;

    template<class Base1, class Base2>
    class LinearInheritFork
        : public Base1
        , public Base2
    {
    };


    /** Rule if head is a class without Base template parameter
     *
     * Create a fork and inherit from head and combined classes from Vec
     */
    template<class Head, class Vec, bool isVectorEmpty = bmpl::empty<Vec>::value>
    struct TypelistLinearInherit;

    template<class Head, class Vec>
    struct TypelistLinearInherit<Head, Vec, false>
    {
        typedef LinearInheritFork<Head, typename LinearInherit<Vec>::type> type;
    };


    /** Rule if head is a class which can inherit from other class
     */
    template<template<class> class Head, class Vec>
    struct TypelistLinearInherit<Head<pmacc::NullFrame>, Vec, false>
    {
        typedef Head<typename LinearInherit<Vec>::type> type;
    };


    /** Rule if Vec is empty but Head is valid
     *
     * This is the recursive end rule
     */
    template<class Head, class Vec>
    struct TypelistLinearInherit<Head, Vec, true>
    {
        typedef Head type;
    };


    /** Create a data structure which inherit linearly
     * \tparam vec_ boost mpl vector with classes
     *
     * class A<pmacc::NullFrame>;
     * LinearInherit<mpl::vector<A<>,B> >::type return
     *
     * typedef A<B> type;
     */
    template<typename vec_>
    struct LinearInherit
    {
        typedef
            typename TypelistLinearInherit<typename bmpl::front<vec_>::type, typename bmpl::pop_front<vec_>::type>::
                type type;
    };

} // namespace pmacc
