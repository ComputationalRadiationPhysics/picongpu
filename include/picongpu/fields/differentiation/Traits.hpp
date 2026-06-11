/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/differentiation/Derivative.def"

#include <cstdint>

namespace picongpu::fields::differentiation::traits
{
    /** Type trait for derivative functor for the given derivative tag and
     *  direction, accessible as ::type
     *
     * Has to be specialized for each derivative tag.
     *
     * @tparam T_Derivative derivative tag, defines the finite-difference scheme
     * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
     */
    template<typename T_Derivative, uint32_t T_direction>
    struct DerivativeFunctor;
} // namespace picongpu::fields::differentiation::traits
