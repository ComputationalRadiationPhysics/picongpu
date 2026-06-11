/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "Vector.hpp"

namespace pmacc
{
    namespace math
    {
        template<uint32_t dim>
        using Size_t = Vector<size_t, dim>;
    } // namespace math
} // namespace pmacc
