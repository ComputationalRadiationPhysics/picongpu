/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/identifier/identifier.hpp"
#include "pmacc/ppFunctions.hpp"
#include "pmacc/traits/Resolve.hpp"
#include "pmacc/types.hpp"

#include <string>
#include <type_traits>

namespace pmacc
{
    identifier(pmacc_void, );
    identifier(pmacc_isAlias, );
} // namespace pmacc

/** create an alias
 *
 * an alias is a unspecialized type of an identifier or a value_identifier
 *
 * @param name name of alias
 *
 * example: alias(aliasName); //create type varname
 *
 * to specialize an alias do: aliasName<valueIdentifierName>
 * to create an instance of this alias you can use:
 *      aliasName();   or aliasName_
 *
 * get type which is represented by the alias
 *      typedef typename traits::Resolve<name>::type resolved_type;
 */
#define alias(name)                                                                                                   \
    template<typename T_Type = pmacc::pmacc_void, typename T_IsAlias = pmacc::pmacc_isAlias>                          \
    struct name                                                                                                       \
    {                                                                                                                 \
        static std::string getName()                                                                                  \
        {                                                                                                             \
            return std::string(#name);                                                                                \
        }                                                                                                             \
    };                                                                                                                \
    constexpr name<> PMACC_JOIN(name, _)

namespace pmacc
{
    namespace traits
    {
        template<template<typename, typename> class T_Object, typename T_AnyType>
        struct Resolve<T_Object<T_AnyType, pmacc::pmacc_isAlias>>
        {
            /*solve recursive if alias is nested*/
            using type = mp_if<
                std::is_same<T_AnyType, typename Resolve<T_AnyType>::type>,
                T_AnyType,
                typename Resolve<T_AnyType>::type>;
        };

    } // namespace traits
} // namespace pmacc
