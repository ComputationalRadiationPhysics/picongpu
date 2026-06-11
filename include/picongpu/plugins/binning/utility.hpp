/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/memory/tuple/utility.hpp>

#include <tuple>
#include <utility>

namespace picongpu
{
    namespace plugins::binning
    {
        template<typename... Args>
        constexpr auto createTuple(Args&&... args)
        {
            return std::make_tuple(std::forward<Args>(args)...);
        }

        template<template<typename...> typename TypeTemplate, typename... Args>
        auto make_unique(Args&&... args)
        {
            auto ptr = new TypeTemplate(std::forward<Args>(args)...);
            return std::unique_ptr<std::remove_cvref_t<decltype(*ptr)>>(ptr);
        }

    } // namespace plugins::binning
} // namespace picongpu
