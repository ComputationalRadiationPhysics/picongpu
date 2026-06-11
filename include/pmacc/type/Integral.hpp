/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace pmacc
{
    namespace type
    {
        using id_t = uint64_t;
        using uint64_cu = unsigned long long int;
        using int64_cu = long long int;

    } // namespace type

    // for backward compatibility pull all definitions into the pmacc namespace
    using namespace type;
} // namespace pmacc
