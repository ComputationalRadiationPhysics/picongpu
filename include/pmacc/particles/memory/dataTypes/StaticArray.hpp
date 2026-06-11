/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

#include <cstdint>

namespace pmacc
{
    template<typename T_Type, typename T_size>
    class StaticArray
    {
    public:
        static constexpr uint32_t size = T_size::value;
        using Type = T_Type;

    private:
        Type data[size];

    public:
        HDINLINE
        Type& operator[](int const idx)
        {
            return data[idx];
        }

        HDINLINE
        Type const& operator[](int const idx) const
        {
            return data[idx];
        }
    };

} // namespace pmacc
