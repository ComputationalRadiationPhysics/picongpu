/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/particles/memory/frames/NullFrame.hpp"
#include "pmacc/types.hpp"

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
    template<class Head, class Vec, bool isVectorEmpty = mp_empty<Vec>::value>
    struct TypelistLinearInherit;

    template<class Head, class Vec>
    struct TypelistLinearInherit<Head, Vec, false>
    {
        using type = LinearInheritFork<Head, typename LinearInherit<Vec>::type>;
    };

    /** Rule if head is a class which can inherit from other class
     */
    template<template<class> class Head, class Vec>
    struct TypelistLinearInherit<Head<pmacc::NullFrame>, Vec, false>
    {
        using type = Head<typename LinearInherit<Vec>::type>;
    };

    /** Rule if Vec is empty but Head is valid
     *
     * This is the recursive end rule
     */
    template<class Head, class Vec>
    struct TypelistLinearInherit<Head, Vec, true>
    {
        using type = Head;
    };

    /** Create a data structure which inherit linearly
     * @tparam vec_ boost mpl vector with classes
     *
     * class A<pmacc::NullFrame>;
     * LinearInherit<mpl::vector<A<>,B> >::type return
     *
     * typedef A<B> type;
     */
    template<typename vec_>
    struct LinearInherit
    {
        using type = typename TypelistLinearInherit<mp_front<vec_>, mp_pop_front<vec_>>::type;
    };

} // namespace pmacc
