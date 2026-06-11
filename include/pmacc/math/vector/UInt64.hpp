/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Axel Huebl
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
        using UInt64 = Vector<uint64_t, dim>;
    } // namespace math
} // namespace pmacc
