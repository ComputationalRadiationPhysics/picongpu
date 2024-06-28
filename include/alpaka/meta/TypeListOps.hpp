/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <tuple>
#include <type_traits>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename List>
        struct Front
        {
        };

        template<template<typename...> class List, typename Head, typename... Tail>
        struct Front<List<Head, Tail...>>
        {
            using type = Head;
        };
    } // namespace detail

    template<typename List>
    using Front = typename detail::Front<List>::type;

    template<typename List, typename Value>
    struct Contains : std::false_type
    {
    };

    template<template<typename...> class List, typename Head, typename... Tail, typename Value>
    struct Contains<List<Head, Tail...>, Value>
    {
        static constexpr bool value = std::is_same_v<Head, Value> || Contains<List<Tail...>, Value>::value;
    };

    // copied from https://stackoverflow.com/a/51073558/22035743
    template<typename T>
    struct IsList : std::false_type
    {
    };

    template<template<typename...> class TList, typename... TTypes>
    struct IsList<TList<TTypes...>> : std::true_type
    {
    };

    //! \brief Checks whether the specified type is a list. List is a type with a variadic number of template types.
    template<typename T>
    constexpr bool isList = IsList<std::decay_t<T>>::value;

    namespace detail
    {
        template<template<typename...> class TListType, typename TType, typename = void>
        struct ToListImpl
        {
            using type = TListType<TType>;
        };

        template<template<typename...> class TListType, typename TList>
        struct ToListImpl<TListType, TList, std::enable_if_t<alpaka::meta::isList<TList>>>
        {
            using type = TList;
        };
    } // namespace detail

    //! \brief Takes an arbitrary number of types (T) and creates a type list of type TListType with the types (T). If
    //! T is a single template parameter and it satisfies alpaka::meta::isList, the type of the structure is T (no type
    //! change). For example std::tuple can be used as TListType.
    //! \tparam TListType type of the created list
    //! \tparam T possible list types or type list
    template<template<typename...> class TListType, typename... T>
    struct ToList;

    template<template<typename...> class TListType, typename T>
    struct ToList<TListType, T> : detail::ToListImpl<TListType, T>
    {
    };

    template<template<typename...> class TListType, typename T, typename... Ts>
    struct ToList<TListType, T, Ts...>
    {
        using type = TListType<T, Ts...>;
    };

    //! \brief If T is a single argument and a type list (fullfil alpaka::meta::isList), the return type is T.
    //! Otherwise, std::tuple is returned with T types as template parameters.
    template<typename... T>
    using ToTuple = typename ToList<std::tuple, T...>::type;


} // namespace alpaka::meta
