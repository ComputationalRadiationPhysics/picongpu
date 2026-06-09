/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2025 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/math/math.hpp"
#include "pmacc/math/vector/Vector.hpp"

#include <cmath>
#include <concepts>

namespace pmacc::math
{

    // type specific default tolerances
    template<std::floating_point T>
    struct DefaultTolerances;

    template<>
    struct DefaultTolerances<float>
    {
        static constexpr float rtol = 1.0e-5f;
        static constexpr float atol = 1.0e-8f;
    };

    template<>
    struct DefaultTolerances<double>
    {
        static constexpr double rtol = 1.0e-9;
        static constexpr double atol = 1.0e-12;
    };

    /**
     * @brief Checks if two floating point numbers are approximately equal
     * @details Implements the check: abs(a - b) <= (atol + rtol * max(abs(a), abs(b)))
     * For non-finite numbers: inf == inf, nan != anything
     * @tparam T Floating-point type
     * @param a Value to compare
     * @param b Value to compare
     * @param rtol Relative tolerance
     * @param atol Absolute tolerance
     * @return True if the values are approximately equal, false otherwise.
     */
    template<std::floating_point T>
    constexpr bool isApproxEqual(T a, T b, T rtol = DefaultTolerances<T>::rtol, T atol = DefaultTolerances<T>::atol)
    {
        if(!std::isfinite(a) || !std::isfinite(b))
        {
            return a == b;
        }

        return pmacc::math::abs(a - b) <= (atol + rtol * pmacc::math::max(pmacc::math::abs(a), pmacc::math::abs(b)));
    }

    /**
     * @brief Checks if a floating point number is approximately equal to zero
     * @tparam T Floating-point type
     * @param value Value to check
     * @param atol Absolute tolerance
     * @return True if the value is approximately zero, false otherwise
     */
    template<std::floating_point T>
    constexpr bool isApproxZero(T value, T atol = DefaultTolerances<T>::atol)
    {
        return pmacc::math::abs(value) <= atol;
    }

    /** @brief Checks if two vectors are approximately equal element-wise
     * @tparam T Floating-point element type
     * @tparam T_dim Vector dimension
     */
    template<std::floating_point T, uint32_t T_dim>
    constexpr bool isApproxEqual(
        Vector<T, T_dim> const& a,
        Vector<T, T_dim> const& b,
        T rtol = DefaultTolerances<T>::rtol,
        T atol = DefaultTolerances<T>::atol)
    {
        for(uint32_t i = 0u; i < T_dim; ++i)
            if(!isApproxEqual(a[i], b[i], rtol, atol))
                return false;
        return true;
    }

} // namespace pmacc::math
