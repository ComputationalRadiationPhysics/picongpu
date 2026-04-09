/* Copyright 2023-2024 Tapish Narwal
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
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
