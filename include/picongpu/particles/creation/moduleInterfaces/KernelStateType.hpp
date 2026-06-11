/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of KernelState type
     *
     * The particle creation kernel has a between threads shared state in shared memory.
     * This shared state is defined by this struct, which is initialized by the InitKernelStateFunctor
     */
    template<typename... T_KernelConfigOptions>
    struct KernelStateType;
} // namespace picongpu::particles::creation::moduleInterfaces
