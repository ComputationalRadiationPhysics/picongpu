/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace pmacc::exec::detail
{
    /** Object to launch the kernel functor on the device.
     *
     * This objects contains the kernel functor, kernel meta information and the launch parameters.
     * Object is used to enqueue the kernel with user arguments on the device.
     *
     * @tparam T_Kernel pmacc Kernel object
     */
    template<typename T_Kernel, uint32_t T_dim>
    struct KernelLauncher;
} // namespace pmacc::exec::detail
