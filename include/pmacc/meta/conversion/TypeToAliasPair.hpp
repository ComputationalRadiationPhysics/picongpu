/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/identifier/alias.hpp"
#include "pmacc/meta/Pair.hpp"
#include "pmacc/meta/conversion/TypeToPair.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** create pmacc::meta::Pair
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
        using type = typename TypeToPair<T_Type>::type;
    };

    /** specialisation if T_Type is a pmacc alias*/
    template<template<typename, typename> class T_Alias, typename T_Type>
    struct TypeToAliasPair<T_Alias<T_Type, pmacc::pmacc_isAlias>>
    {
        using type
            = pmacc::meta::Pair<T_Alias<pmacc_void, pmacc::pmacc_isAlias>, T_Alias<T_Type, pmacc::pmacc_isAlias>>;
    };


} // namespace pmacc
