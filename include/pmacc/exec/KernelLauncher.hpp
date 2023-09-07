/* Copyright 2022-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


namespace pmacc::exec::detail
{
    /** Object to launch the kernel functor on the device.
     *
     * This objects contains the kernel functor, kernel meta information and the launch parameters.
     * Object is used to enqueue the kernel with user arguments on the device.
     *
     * @tparam T_Kernel pmacc Kernel object
     */
    template<typename T_Kernel>
    struct KernelLauncher;
} // namespace pmacc::exec::detail
