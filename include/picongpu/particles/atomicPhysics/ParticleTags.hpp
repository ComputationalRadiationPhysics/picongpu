/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/HasFlag.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    struct Tags
    {
        struct Ion
        {
        };

        struct Electron
        {
        };

        struct OnlyIPDIon
        {
        };

        struct OnlyIPDElectron
        {
        };
    };
} // namespace picongpu::particles::atomicPhysics
