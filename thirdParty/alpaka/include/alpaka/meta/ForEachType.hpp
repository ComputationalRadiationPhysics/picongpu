/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <utility>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename TList>
        struct ForEachTypeHelper;

        template<template<typename...> class TList>
        struct ForEachTypeHelper<TList<>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TFnObj, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto forEachTypeHelper(TFnObj&& /* f */, TArgs&&... /* args */) -> void
            {
            }
        };

        template<template<typename...> class TList, typename T, typename... Ts>
        struct ForEachTypeHelper<TList<T, Ts...>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TFnObj, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto forEachTypeHelper(TFnObj&& f, TArgs&&... args) -> void
            {
                f.template operator()<T>(std::forward<TArgs>(args)...);
                ForEachTypeHelper<TList<Ts...>>::forEachTypeHelper(
                    std::forward<TFnObj>(f),
                    std::forward<TArgs>(args)...);
            }
        };
    } // namespace detail

    //! Equivalent to boost::mpl::for_each but does not require the types of the sequence to be default
    //! constructible. This function does not create instances of the types instead it passes the types as template
    //! parameter.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TList, typename TFnObj, typename... TArgs>
    ALPAKA_FN_HOST_ACC auto forEachType(TFnObj&& f, TArgs&&... args) -> void
    {
        detail::ForEachTypeHelper<TList>::forEachTypeHelper(std::forward<TFnObj>(f), std::forward<TArgs>(args)...);
    }
} // namespace alpaka::meta
